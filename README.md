# Test

python main.py --config local_agent_card.yaml --serve --host localhost --port 5000

curl http://localhost:5000/.well-known/agent-card.json                                                                                                     
# add more providers
langchain_openai.ChatOpenAI  
langchain_anthropic.ChatAnthropic  

# external memory
To have a persistant memory, replace ImMemoryStore() with the external database 

# Tool
langchain_mcp_adapters.client.MultiServerMCPClient gives the chance to utilize multiple tools at the same time



