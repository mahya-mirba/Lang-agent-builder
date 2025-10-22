import logging
from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutor
from a2a.server.events import EventQueue
from a2a.server.agent_execution import RequestContext
from a2a.types import InternalError, TaskState, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from a2a.server.tasks import TaskUpdater
from agent import RootAgent

logger = logging.getLogger(__name__)

class LangGraphA2AExecutor(A2AAgentExecutor):
    """A2A-compliant executor for LangGraph agents."""
    
    def __init__(self, agent: RootAgent):
        self.agent = agent
        
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the agent following A2A protocol."""
        try:
            # Extract the user's message
            query = context.get_user_input()
            task = context.current_task
            
            if not task:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)
            
          
            updater = TaskUpdater(event_queue, task.id, task.context_id)
            
            # Use context ID as thread ID for continuity
            thread_id = task.context_id if task else "default"
            user_id = getattr(context, 'user_id', "anon")
            
            # Invoke our agent
            result = await self.agent.invoke(query, thread_id=thread_id, user_id=user_id)
            
            # Extract the response
            response_key = next((key for key in result.keys() if key.endswith('_response')), None)
            output = result.get(response_key, "") if response_key else str(result)
            
            # Create message to send back through A2A
            message = new_agent_text_message(
                output,
                context_id=task.context_id,
                task_id=task.id,
            )
            
            # Update task status 
            await updater.update_status(TaskState.completed, message, final=True)
            
        except Exception as e:
            logger.exception("Error executing agent")
            raise ServerError(error=InternalError()) from e
    
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())