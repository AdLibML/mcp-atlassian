import os
import logging
import warnings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_mcp_tools import convert_mcp_to_langchain_tools
from langchain_ollama import ChatOllama  # Commented out Ollama
from langchain_openai import ChatOpenAI  # Added OpenAI
import asyncio
from load_dotenv import load_dotenv

from langchain_core.messages import AIMessage

def last_ai_text(run_output) -> str | None:
    """Return the content of the last AI message in a LangGraph run output."""
    # Handle both dict outputs and objects with .messages
    messages = (
        run_output.get("messages")
        if isinstance(run_output, dict)
        else getattr(run_output, "messages", None)
    )
    if not messages:
        return None

    # Walk backwards to skip any ToolMessage/FunctionMessage tails
    for m in reversed(messages):
        # Works for AIMessage or generic message with .type == "ai" / role == "assistant"
        if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai" or getattr(m, "role", "") == "assistant":
            return getattr(m, "content", None)
    return None

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

logger = get_logger("agent")


# Suppress Windows pipe cleanup warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
os.environ["PYTHONWARNINGS"] = "ignore::ResourceWarning"

load_dotenv()

mode = os.environ.get("MODE", "local")
mode = 'local'
logger.info(f"Running in {mode} mode")

# Switch to OpenAI GPT-5 for better tool usage
model = ChatOllama(model="qwen3:8b", temperature=0.1, streaming=True,
                   base_url="http://localhost:11434")

model = ChatOpenAI(
    model="gpt-4.1",  # Using GPT-5
    temperature=0.1,
    streaming=True,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
# logger.info(f"Using model: {model.model}")

if mode == "local":
    raw_transport = os.environ.get("TRANSPORT", "streamable-http")
    atlassian_transport = (
        "streamable_http"
        if raw_transport in {"streamable-http", "streamable_http"}
        else raw_transport
    )
    atlassian_default_path = (
        "/mcp" if atlassian_transport == "streamable_http" else "/sse"
    )
    atlassian_default_url = (
        f"http://localhost:{os.environ.get('PORT', '10000')}{atlassian_default_path}"
    )
    server_params = {
        # "math": {
        #     "command": "python",
        #     "args": ["C:/Users/garyj/Code/mcpserver/src/servers/math_server.py"],
        #     "transport": "stdio",
        # },
        # "weather": {
        #     "command": "python", 
        #     "args": ["C:/Users/garyj/Code/mcpserver/src/servers/weather_server.py"],
        #     "transport": "stdio",
        # },
        # "brave": {
        #     "command": "python",
        #     "args": ["C:/Users/garyj/Code/mcpserver/src/servers/brave_server.py"],
        #     "transport": "stdio",
        # },
        # "spotify": {
        #     "command": "uv",
        #     "args": [
        #         "--directory",
        #         "C:/Users/garyj/Code/spotify-mcp",
        #         "run",
        #         "spotify-mcp"
        #     ],
        #     "env": {
        #         "SPOTIFY_CLIENT_ID": os.getenv("SPOTIFY_CLIENT_ID"),
        #         "SPOTIFY_CLIENT_SECRET": os.getenv("SPOTIFY_CLIENT_SECRET"),
        #         "SPOTIFY_REDIRECT_URI": os.getenv("SPOTIFY_REDIRECT_URI")
        #     },
        #     "transport": "stdio",
        # },
        "atlassian": {
            "url": os.environ.get("ATLASSIAN_URL", atlassian_default_url),
            "transport": atlassian_transport,
        }
    }
else:
    raw_transport = os.environ.get("TRANSPORT", "streamable-http")
    atlassian_transport = (
        "streamable_http"
        if raw_transport in {"streamable-http", "streamable_http"}
        else raw_transport
    )
    atlassian_default_path = (
        "/mcp" if atlassian_transport == "streamable_http" else "/sse"
    )
    server_params = {
        "math": {
            "url": os.environ.get("MATH_URL", "http://localhost:5001/mcp/sse"),
            "transport": "sse",
        },
        "weather": {
            "url": os.environ.get("WEATHER_URL", "http://localhost:5000/mcp/sse"),
            "transport": "sse",
        },
        "brave": {
            "url": os.environ.get("BRAVE_URL", "http://localhost:5002/mcp/sse"),
            "transport": "sse",
        },
        "atlassian": {
            "url": os.environ.get(
                "ATLASSIAN_URL",
                f"http://localhost:{os.environ.get('PORT', '10000')}{atlassian_default_path}",
            ),
            "transport": atlassian_transport,
        }
        # Add Spotify SSE URL if you have a production deployment
    }

async def validate_tools(client):
    """Validate that tools are properly loaded and accessible."""
    try:
        tools = await client.get_tools()
        logger.info(f"Successfully loaded {len(tools)} tools")
        
        # Log available tool names for debugging
        tool_names = [tool.name for tool in tools]
        logger.info(f"Available tools: {tool_names}")
        
        # Check for Atlassian-specific tools
        atlassian_tools = [name for name in tool_names if 'jira' in name.lower() or 'confluence' in name.lower()]
        logger.info(f"Atlassian tools found: {atlassian_tools}")
        
        # Log detailed information about Atlassian tools
        for tool in tools:
            if 'jira' in tool.name.lower() or 'confluence' in tool.name.lower():
                logger.info(f"Tool: {tool.name}")
                logger.info(f"  Description: {tool.description}")
                # Fix the schema access - check if it's a dict or has schema method
                if hasattr(tool, 'args_schema'):
                    if hasattr(tool.args_schema, 'schema'):
                        logger.info(f"  Parameters: {tool.args_schema.schema()}")
                    elif isinstance(tool.args_schema, dict):
                        logger.info(f"  Parameters: {tool.args_schema}")
                    else:
                        logger.info(f"  Parameters: {type(tool.args_schema)}")
        
        return tools
    except Exception as e:
        logger.error(f"Error loading tools: {e}")
        raise

async def main(query: str):
    logger.info("Agent starting with query: %s", query)
    
    # Create client without context manager
    client = MultiServerMCPClient(server_params)
    
    try:
        # Validate tools before creating agent
        tools = await validate_tools(client)
        
        # Create agent with validated tools
        agent = create_react_agent(model, tools)
        
        # Add tool usage logging
        logger.info(f"Agent created with {len(tools)} tools")
        
        # Create a system message to guide the model
        system_message = """You are an AI assistant that helps users interact with JIRA and Confluence through MCP tools.

IMPORTANT: When using JIRA tools, use these parameter names:
- Use 'limit' instead of 'maxResults' 
- Use 'start_at' instead of 'start'
- Use 'jql' for JIRA queries
- Use 'query' for Confluence searches

IMPORTANT: Use the correct tool names:
- For JIRA search: use 'jira_search' (not 'jira_search_issues')
- For Confluence search: use 'confluence_search'

Examples:
- jira_search: {"jql": "project = PROJ AND status = 'Backlog'", "limit": 10, "start_at": 0}
- confluence_search: {"query": "documentation", "limit": 10}

Always use the correct parameter names as defined in the tool schemas."""
        
        # Add the system message to the query
        enhanced_query = f"{system_message}\n\nUser query: {query}"
        
        response = await agent.ainvoke({"messages": enhanced_query})
        logger.info("Agent received response.")
        
        # Log the response for debugging
        logger.info(f"Response type: {type(response)}")
        if hasattr(response, 'messages'):
            logger.info(f"Response messages: {len(response.messages)}")
            
    finally:
        # Clean up the client - remove aclose() since it doesn't exist
        # The client will be cleaned up automatically
        pass
            
    return response

if __name__ == "__main__":
    # Test queries for Spotify
    # query = "please put some runescape music on Spotify and play them in the waiting list"
    # query = "Play my favorite songs on Spotify"
    # query = "Search for 'The Beatles' on Spotify"
    # query = "Please get the in progress tickets from JIRA."
    # query = "Please tell me from confluence what the project is about."
    # query = "Get all the projects from Jira."
    query = "can you comment the stories in progress a way to solve the stuff and a stuctured strategy to do it."
    query = "please extract and tell me the roadmap of the project from confluence."
    # query ="from the jira tickets, what should i do first ?"
    query = "tell me the names of the confluence spaces."

    response = asyncio.run(main(query))
    print(response)
    print(last_ai_text(response))
