import logging
import asyncio
from uuid import uuid4
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    
    base_url = 'http://localhost:5000'
    
    async with httpx.AsyncClient() as httpx_client:
        # Fetch agent card
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        logger.info(f"Found agent card: {agent_card.name}")
        
        
        agent_card.url = base_url
        
        
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        
        # Send a test message
        payload = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': 'Hello, what can you help me with?'}],
                'message_id': uuid4().hex,
            },
        }
        
        request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))
        logger.info(f"Sending request to {agent_card.url}")
        response = await client.send_message(request)
        
        print(response.model_dump(mode='json', exclude_none=True))

if __name__ == '__main__':
    asyncio.run(main())