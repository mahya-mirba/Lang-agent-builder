from typing import Annotated, Literal, Dict, Any

from a2a.types import AgentCapabilities, AgentCard, AgentProvider, AgentSkill
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    RootModel,
    TypeAdapter,
    ValidationInfo,
    field_validator,
)


SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'application/pdf']


# --------------------------------A2A Agent Card--------------------------------#
class DefaultAgentCard(AgentCard):
    """Defines Default AgentCard."""

    name: str | None = None
    description: str | None = None
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    default_input_modes: list[str] = SUPPORTED_CONTENT_TYPES
    default_output_modes: list[str] = SUPPORTED_CONTENT_TYPES
    skills: list[AgentSkill] = Field(default_factory=list)
    provider: AgentProvider = AgentProvider(
        organization='capgemini', url='https://www.capgemini.com/'
    )
    version: str = Field(default='1.0.0')
    url: str = Field(default="http://localhost:5000")


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxA2A Agent Cardxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#


class MCPToolConfig(BaseModel):
    """Defines Tool Config."""

    type: Literal['mcp'] = 'mcp'
    url: str

    @field_validator('url')
    @classmethod
    def validate_tool_url(cls, v: str | None) -> str | None:
        """Validates url of the mcp tool."""
        if v is not None:
            try:
                HttpUrl(v)  # Validate as HttpUrl
            except Exception as e:  # Catch validation errors
                raise ValueError(
                    f"Invalid URL format for ToolConfig.url '{v}': {e}"
                ) from e
        return v  # Return original string or None if valid


class BaseAgentModel(BaseModel):
    """Defines Base config of an agent."""

    name: str
    type: str
    description: str


class LlmAgentModel(BaseAgentModel):
    """Defines config of an LLM agent."""

    type: Literal['llm']
    name: str | None = None
    model: str | None = Field(default=None)
    instructions: str | list[str] | None = Field(
        default=None, validate_default=True
    )
    tool_use: bool = False

class llm_config(BaseModel):
    """Defines LLM configuration."""
    

    google_api_key: str | None = Field(
        default=None,
        description="API key for Google Generative AI services."
    )
    openai_api_key: str | None = Field(
        default=None,
        description="API key for OpenAI services."
    )   
    
class A2ARemoteAgentModel(BaseAgentModel):
    """Defines config of a A2A Remote agent."""

    type: Literal['remote']
    name: str | None = None
    url: str
    description: str | None = None


# A different approach to handling the nested discrimination
class AgentsConfig(BaseModel):
    """Defines and validates structure of agents list as a key-value pair."""

    root: Dict[str, Dict[str, Any]]  # Changed to accept any dict first
    
    @field_validator('root')
    @classmethod
    def validate_agent_graph(cls, agents_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validates agent graph and converts to proper agent types."""
        if not agents_dict:
            raise ValueError("The 'agents' configuration cannot be empty.")
        
        validated_agents = {}
        
        for agent_name, agent_spec in agents_dict.items():
            # Ensure agent_name gets into the spec
            agent_spec['name'] = agent_name
            
            # Check the agent type and validate
            agent_type = agent_spec.get('type')
            if not agent_type:
                raise ValueError(f"Agent '{agent_name}' has no 'type' specified")
            
            if agent_type == 'llm':
                validated_agent = LlmAgentModel(**agent_spec)
            elif agent_type == 'remote':
                validated_agent = A2ARemoteAgentModel(**agent_spec)
            else:
                raise ValueError(f"Agent type '{agent_type}' is not supported")
                
            validated_agents[agent_name] = validated_agent
            
        return validated_agents


class AppConfig(BaseModel):
    """Defines structure of App configs."""

    app_name: str
    version: str = '1.0.0'
    tools: MCPToolConfig | None = None
    framework: Literal['adk', 'langchain', 'langgraph'] = Field(default='langgraph')
    agent_card: DefaultAgentCard
    agents: AgentsConfig
    root_agent: str | None = None

    model_config = {'extra': 'forbid', 'validate_assignment': True}

    @field_validator('agents')
    @classmethod
    def validate_tool_usage_consistency(
        cls, v: AgentsConfig, info: ValidationInfo
    ) -> AgentsConfig:
        """Validates the tools usage."""
        tools_config = info.data.get('tools')
        for agent_name, agent_spec in v.root.items():
            if (
                isinstance(agent_spec, LlmAgentModel)
                and agent_spec.tool_use
                and not tools_config
            ):
                raise ValueError(
                    f"Agent '{agent_name}' is configured with 'tool_use: true', "
                    "but no 'tools' section is defined in the application configuration."
                )
        return v