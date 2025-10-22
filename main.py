import argparse
import asyncio
import dataclasses
import logging
import os
from typing import Any, Dict

import httpx
import uvicorn
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from agent_executor import LangGraphA2AExecutor
from agent import RootAgent
from config_resolvers.utils import get_resolver


def is_prod() -> bool:
    """Checks whether env is production."""
    return os.getenv('ENVIRONMENT', None) == 'production'


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.INFO))


@dataclasses.dataclass
class ServerOpts:
    """Holds the server options."""
    host: str
    port: int


async def cmd_app(agent: RootAgent) -> None:
    """Start the agent for commandline interaction."""
    print(f"Starting CLI interaction with {agent.app_name} agent...")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() == 'exit':
                print("Exiting...")
                break
                
            # Process the input through the agent
            result = await agent.invoke(user_input)
            
            # Extract the response
            response_key = next((key for key in result.keys() if key.endswith('_response')), None)
            if response_key:
                print(f"\nAgent: {result[response_key]}")
            else:
                print(f"\nAgent: {result}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


DEFAULT_TASK_STORE = InMemoryTaskStore()


def build_fastapi(agent: RootAgent) -> FastAPI:
    """Builds and returns fastapi app for agent."""
    root_agent_name = next(iter(agent.app_config.agents.root.keys()))
    root_agent = agent.app_config.agents.root[root_agent_name]
    agent_card = RootAgent.construct_agent_card(
        root_agent, agent.app_config.agent_card
    )
    
    # Use the A2A-compliant executor
    agent_executor = LangGraphA2AExecutor(agent=agent)
    
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=DEFAULT_TASK_STORE
    )
    a2a_app = A2AFastAPIApplication(
        agent_card=agent_card, http_handler=request_handler
    )
    
    app = a2a_app.build()
    
    # Keep your custom endpoints
    @app.post("/generate")
    async def generate(request: Request) -> JSONResponse:
        data = await request.json()
        input_text = data.get("input", "")
        thread_id = data.get("thread_id", "default")
        user_id = data.get("user_id", "anon")
        
        result = await agent.invoke(input_text, thread_id=thread_id, user_id=user_id)
        
        response_key = next((key for key in result.keys() if key.endswith('_response')), None)
        output = result.get(response_key, "") if response_key else str(result)
        
        return JSONResponse({"output": output})
    
    return app


async def a2a_server(agent: RootAgent, opts: ServerOpts) -> None:
    """Starts the a2a server for the agent."""
    app = build_fastapi(agent=agent)
    server = uvicorn.Server(
        uvicorn.Config(app=app, host=opts.host, port=opts.port)
    )
    await server.serve()


async def start_app(
    config: dict[str, Any],
    to_serve: bool = False,
    srv_opts: ServerOpts | None = None,
) -> None:
    """Starts the app as defined."""
    try:
        agent = RootAgent(config)
        async with httpx.AsyncClient(
            timeout=120, verify=is_prod()
        ) as httpx_client:
            await agent.build(httpx_client=httpx_client)
            if to_serve:
                return await a2a_server(agent=agent, opts=srv_opts)
            return await cmd_app(agent=agent)
    except asyncio.CancelledError:
        logging.info('Shutting down the a2a server')


def main() -> None:
    """Entrypoint of the application."""
    parser = argparse.ArgumentParser(
        'Agent Builder', description='Builder for Agents'
    )
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--src', default='local', choices=['local', 'gcs'], help='Source type')
    parser.add_argument('--serve', action='store_true', help='Run in server mode')
    parser.add_argument('--host', default='localhost', help='Host to serve on')
    parser.add_argument('--port', default='5000', help='Port to serve on')
    args = parser.parse_args()
    srv_opts = None
    if args.serve:
        srv_opts = ServerOpts(host=args.host, port=int(args.port))

    resolver = get_resolver(args.src)
    config = resolver.load(path=args.config)
    asyncio.run(
        start_app(config=config, to_serve=args.serve, srv_opts=srv_opts)
    )


if __name__ == '__main__':
    main()