import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Dict, List, Optional, Union
import httpx
from a2a.client.card_resolver import A2ACardResolver
from a2a.types import AgentCard
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from typing_extensions import TypedDict, Annotated
from operator import add

from schema import (
    A2ARemoteAgentModel,
    AppConfig,
    BaseAgentModel,
    DefaultAgentCard,
    LlmAgentModel,
)
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL = "gemini-2.5-flash"
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[List[Any], add]  # accumulate across steps
    content: str

class ToolBuilder:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self._mcp_client: MultiServerMCPClient | None = None
        self.lc_tools: List[BaseTool] | None = None

    async def _build_mcp_tools(self) -> None:
        if not getattr(self.app_config, "tools", None):
            logger.info("No tools configuration found")
            return

        url = str(self.app_config.tools.url)
        logger.info("Building MCP tools from url -> %s", url)
        self._mcp_client = MultiServerMCPClient(
            {"remote": {"transport": "streamable_http", "url": url}}
        )
        async with self._mcp_client.session("remote") as session:
            self.lc_tools = await load_mcp_tools(session)
            logger.info(
                "Done initializing MCP tools: %d tools available",
                len(self.lc_tools or []),
            )

    async def get_tools(self) -> List[BaseTool]:
        if self.lc_tools is None:
            await self._build_mcp_tools()
        return self.lc_tools or []

    async def close(self):
        if self._mcp_client:
            logger.info("Closing MCP client")

class RootAgent:
    app_name: str
    root_agent: Runnable
    graph: StateGraph
    tool_builder: ToolBuilder | None = None
    mcp_tools: list[BaseTool] | None = None
    _exit_stack: AsyncExitStack | None = None
    _shared_httpx_client: httpx.AsyncClient | None = None
    app_config: AppConfig

    memory_store: InMemoryStore

    def __init__(self, config: dict[str, Any]):
        try:
            self.app_config = AppConfig.model_validate(config)
        except Exception as e:
            logger.exception("Configuration Validation Error")
            raise e
        self.app_name = self.app_config.app_name

        # Store persists memories ACROSS threads (namespaced)
        self.memory_store = InMemoryStore()

        self.tool_builder = ToolBuilder(self.app_config)

    @staticmethod
    def construct_agent_card(root_agent: BaseAgentModel, a2a_card_model: DefaultAgentCard) -> AgentCard:
        return AgentCard(
            **a2a_card_model.model_dump(exclude_none=True),
        )

    async def _build_mcp_tools(self) -> None:
        if self.tool_builder:
            await self.tool_builder._build_mcp_tools()
            self.mcp_tools = await self.tool_builder.get_tools()

    async def _build_remote_agent(self, agent_name: str, agent_model: A2ARemoteAgentModel) -> Runnable:
        if not self._shared_httpx_client:
            raise RuntimeError("Shared httpx client not initialized for remote agent construction.")
        if not agent_model.url:
            raise ValueError(f"URL missing for remote agent '{agent_name}'.")

        url = str(agent_model.url)
        card_resolver = A2ACardResolver(httpx_client=self._shared_httpx_client, base_url=url)
        a2a_card = await card_resolver.get_agent_card()

        class RemoteA2ARunnable(Runnable):
            def __init__(self, name, card, url, client):
                self.name = name
                self.card = card
                self.base_url = url
                self.client = client
                self.description = card.description

            async def ainvoke(self, input_data: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
                content = input_data.get("content", "")
                messages = input_data.get("messages", [])

                if not content and messages:
                    for msg in reversed(messages):
                        if isinstance(msg, HumanMessage):
                            content = msg.content
                            break

                resp = await self.client.post(f"{self.base_url}/generate", json={"input": content})
                resp.raise_for_status()
                result = resp.json()
                output = result.get("output", "")

                return {
                    f"{self.name}_response": output,
                    "messages": messages + [AIMessage(content=output)],
                }

        return RemoteA2ARunnable(
            name=a2a_card.name.replace(" ", ""),
            card=a2a_card,
            url=url,
            client=self._shared_httpx_client,
        )

    async def _build_llm_agent(self, agent_name: str, agent_model: LlmAgentModel):
        instruction = agent_model.instructions
        if isinstance(instruction, list):
            instruction = " ".join(instruction)

        #model = agent_model.model if agent_model.model else DEFAULT_MODEL
        model = DEFAULT_MODEL
        google_api_key = GOOGLE_API_KEY
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key)

        tools = []
        if agent_model.tool_use and self.mcp_tools:
            tools = self.mcp_tools
            logger.info("Agent '%s' configured with %d tools", agent_name, len(tools))

        system_prompt = f"""You are an assistant named {agent_name}.
{agent_model.description}

{instruction or ''}"""

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder(variable_name="messages")]
        )

        # Node signature accepts store + config if you want to read/write long-term memories
        async def agent_node(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> Dict[str, Any]:
            messages = state.get("messages", [])

            # OPTIONAL: read last user message and search user memories
            user_id = (config or {}).get("configurable", {}).get("user_id", "anon")
            namespace = (str(user_id), "memories")

            # Example (safe to remove): pull up to 3 related memories
            try:
                mems = store.search(namespace, query=messages[-1].content if messages else "", limit=3)
                mem_text = "\n".join([m.value.get("memory", "") for m in mems if m.value.get("memory")])
            except Exception:
                mem_text = ""

            if tools:
                agent = create_tool_calling_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(
                    agent=agent, tools=tools, return_intermediate_steps=True, verbose=False
                )
                result = await agent_executor.ainvoke({"messages": messages})
                output = result.get("output", "")
            else:
                chain = prompt | llm
                # Inject mem_text into the last user message to give the model context (optional)
                if mem_text and messages:
                    messages = messages + [HumanMessage(content=f"(Relevant memories)\n{mem_text}")]
                result = await chain.ainvoke({"messages": messages})
                output = result.content

            # OPTIONAL: write a new memory (example heuristic)
            if messages:
                try:
                    store.put(namespace, key=None, value={"memory": f"Last user said: {messages[-1].content}"})
                except Exception:
                    pass

            return {f"{agent_name}_response": output, "messages": state["messages"] + [AIMessage(content=output)]}

        return agent_node

    async def _build_root_agent(self) -> None:
        if not self.app_config.agents.root:
            raise ValueError("No agents defined in the configuration.")

        root_agent_name = next(iter(self.app_config.agents.root.keys()))
        agent_model = self.app_config.agents.root[root_agent_name]

        if isinstance(agent_model, A2ARemoteAgentModel):
            self.root_agent = await self._build_remote_agent(root_agent_name, agent_model)
        elif isinstance(agent_model, LlmAgentModel):
            self.root_agent = await self._build_llm_agent(root_agent_name, agent_model)
        else:
            raise ValueError(f"Agent type '{agent_model.type}' is not supported in this non-sub-agent version.")

        builder = StateGraph(AgentState)
        builder.add_node("agent", self.root_agent)
        builder.set_entry_point("agent")
        builder.add_edge("agent", END)

        checkpointer = InMemorySaver()                   # no args
        self.graph = builder.compile(checkpointer=checkpointer, store=self.memory_store)

        logger.info("Root agent graph compiled successfully")

    async def build(self, httpx_client: httpx.AsyncClient) -> "RootAgent":
        self._shared_httpx_client = httpx_client
        await self._build_mcp_tools()
        await self._build_root_agent()
        logger.info("RootAgent %s is built and ready.", self.app_name.upper())
        return self

    async def invoke(self, input_data: Union[str, Dict[str, Any]], *, thread_id: str = "default", user_id: str = "anon") -> Dict[str, Any]:
        # Normalize input -> messages
        if isinstance(input_data, str):
            input_data = {"messages": [HumanMessage(content=input_data)], "content": input_data}
        elif "messages" not in input_data and "content" in input_data:
            input_data["messages"] = [HumanMessage(content=input_data["content"])]

        # Always pass thread_id (and user_id) so memory persists
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        result = await self.graph.ainvoke(input_data, config=config)
        return result

    async def cleanup(self):
        if self.tool_builder:
            await self.tool_builder.close()
        logger.info("RootAgent cleanup completed")
