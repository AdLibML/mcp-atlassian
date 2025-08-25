import os
import logging
import warnings
import json
import re
import asyncio
from typing import Any, Dict, Optional, List

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_mcp_tools import convert_mcp_to_langchain_tools  # (not used directly, but fine to keep)
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from load_dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, StructuredTool


# =========================
# Utils: get last AI text
# =========================
def last_ai_text(run_output) -> str | None:
    messages = (
        run_output.get("messages")
        if isinstance(run_output, dict)
        else getattr(run_output, "messages", None)
    )
    if not messages:
        return None
    for m in reversed(messages):
        if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai" or getattr(m, "role", "") == "assistant":
            return getattr(m, "content", None)
    return None


# =========================
# Logging setup
# =========================
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
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
mode = "local"
logger.info(f"Running in {mode} mode")

# =========================
# LLM setup
# =========================
model = ChatOllama(model="qwen3:8b", temperature=0.1, streaming=True, base_url="http://localhost:11434")
model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.1,
    streaming=True,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# =========================
# MCP server config
# =========================
if mode == "local":
    server_params = {
        "atlassian": {
            "url": os.environ.get("ATLASSIAN_URL", "http://localhost:8000/sse"),
            "transport": "sse",
        }
    }
else:
    server_params = {
        "atlassian": {
            "url": os.environ.get("ATLASSIAN_URL", "http://localhost:8000/sse"),
            "transport": "sse",
        }
    }


# ============================================================
# Tool Guard: normalize/validate/edit/retry per tool call
# ============================================================
def _clamp_limit(p: Dict[str, Any], *, key: str = "limit", min_v=1, max_v=50):
    if key in p:
        try:
            p[key] = int(p[key])
        except Exception:
            p[key] = 10
        p[key] = max(min_v, min(max_v, p[key]))
    return p


def _normalize_jira_search(p: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(p)
    # Rename legacy keys
    if "maxResults" in q and "limit" not in q:
        q["limit"] = q.pop("maxResults")
    if "start" in q and "start_at" not in q:
        q["start_at"] = q.pop("start")

    _clamp_limit(q)

    # Prefer double quotes for status
    if isinstance(q.get("jql"), str):
        q["jql"] = re.sub(r"status\s*=\s*'([^']+)'", r'status = "\1"', q["jql"])

        # If you use an env project filter, you can auto-inject it here (optional)
        proj_filter = os.getenv("JIRA_PROJECTS_FILTER", "").strip()
        if proj_filter and "project" not in q["jql"]:
            # Project filter may be comma-separated keys
            keys = ",".join(k.strip() for k in proj_filter.split(",") if k.strip())
            if keys:
                q["jql"] = f"project in ({keys}) AND {q['jql']}"

    return q


def _normalize_confluence_search(p: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(p)
    if "cql" in q and "query" not in q:
        q["query"] = q.pop("cql")
    _clamp_limit(q, key="limit", min_v=1, max_v=50)
    return q


def normalize_params(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "jira_search":
        return _normalize_jira_search(params)
    if tool_name == "confluence_search":
        return _normalize_confluence_search(params)
    if tool_name in {
        "jira_get_project_issues",
        "jira_get_board_issues",
        "jira_get_sprint_issues",
        "jira_get_sprints_from_board",
    }:
        return _clamp_limit(dict(params))
    return dict(params)


def auto_repair(tool_name: str, params: Dict[str, Any], err: Exception) -> Optional[Dict[str, Any]]:
    q = dict(params)
    # Small heuristics
    if tool_name == "jira_search":
        if not q.get("jql"):
            q["jql"] = "order by created desc"
            return q
        if "status = '" in q.get("jql", ""):
            q["jql"] = re.sub(r"status\s*=\s*'([^']+)'", r'status = "\1"', q["jql"])
            return q
        _clamp_limit(q)
        return q

    if tool_name == "confluence_search":
        if not q.get("query"):
            q["query"] = 'siteSearch ~ "project"'
            return q
        _clamp_limit(q)
        return q

    return None


async def maybe_interactive_edit(tool_name: str, params: Dict[str, Any], interactive: bool) -> Dict[str, Any]:
    """Optional interactive approval/edit step (non-blocking via to_thread)."""
    if not interactive:
        return params
    try:
        choice = await asyncio.to_thread(
            input,
            f"\n[TOOL {tool_name}] Params:\n{json.dumps(params, indent=2)}\nApprove? [y]/edit/n: ",
        )
        choice = (choice or "y").strip().lower()
        if choice in {"y", ""}:
            return params
        if choice == "n":
            raise RuntimeError(f"User rejected tool call {tool_name}")
        if choice == "edit":
            raw = await asyncio.to_thread(input, "Enter JSON dict for new params: ")
            try:
                new_p = json.loads(raw)
                if isinstance(new_p, dict):
                    return new_p
                print("Not a JSON object; keeping original.")
            except Exception as e:
                print(f"Bad JSON ({e}); keeping original.")
        return params
    except EOFError:
        return params


def wrap_tool(tool: BaseTool, *, interactive: bool = False) -> BaseTool:
    """
    Create a new StructuredTool that proxies to the original tool with
    normalize/validate/retry logic. We do NOT mutate the original StructuredTool.
    """
    original_tool = tool
    tool_name = tool.name
    args_schema = getattr(tool, "args_schema", None)

    async def guard_coroutine(**kwargs):
        params = kwargs if isinstance(kwargs, dict) else dict(kwargs)
        params = normalize_params(tool_name, params)

        # Validate against args_schema if present
        schema_cls = args_schema
        if schema_cls:
            try:
                schema_cls(**params)  # pydantic model call -> validation
            except Exception:
                # attempt a second normalization pass then continue
                params = normalize_params(tool_name, params)

        params = await maybe_interactive_edit(tool_name, params, interactive)

        logger.info(f"[TOOL CALL] {tool_name} -> {params}")

        async def _call_original(p: Dict[str, Any]):
            # Use ainvoke if available; otherwise run sync invoke in a thread
            if hasattr(original_tool, "ainvoke"):
                return await original_tool.ainvoke(p)
            return await asyncio.to_thread(original_tool.invoke, p)

        try:
            result = await _call_original(params)
            logger.info(f"[TOOL OK] {tool_name}")
            return result
        except Exception as e:
            logger.warning(f"[TOOL ERR] {tool_name}: {e}")
            fix = auto_repair(tool_name, params, e)
            if fix is not None:
                logger.info(f"[TOOL RETRY] {tool_name} -> {fix}")
                return await _call_original(fix)
            raise

    # Build a fresh StructuredTool with same name/desc/schema but our guard coroutine
    wrapped = StructuredTool.from_function(
        name=tool_name,
        description=tool.description,
        args_schema=args_schema,     # preserve schema for the agent/tool planner
        coroutine=guard_coroutine,   # our guarded async caller
    )
    return wrapped


def wrap_tools(tools: List[BaseTool], *, interactive: bool = False) -> List[BaseTool]:
    return [wrap_tool(t, interactive=interactive) for t in tools]


# ==========================================
# Load & wrap tools
# ==========================================
def _log_schema(tool: BaseTool):
    schema_cls = getattr(tool, "args_schema", None)
    if not schema_cls:
        logger.info("  Parameters: <none>")
        return
    try:
        # Pydantic v2
        if hasattr(schema_cls, "model_json_schema"):
            logger.info(f"  Parameters: {schema_cls.model_json_schema()}")
        else:
            # v1
            logger.info(f"  Parameters: {schema_cls.schema()}")
    except Exception:
        logger.info("  Parameters: <schema unavailable>")


async def validate_tools(client):
    """Load tools, log schemas, then return wrapped tools."""
    tools = await client.get_tools()
    logger.info(f"Successfully loaded {len(tools)} tools")
    tool_names = [tool.name for tool in tools]
    logger.info(f"Available tools: {tool_names}")

    atlassian_tools = [name for name in tool_names if "jira" in name.lower() or "confluence" in name.lower()]
    logger.info(f"Atlassian tools found: {atlassian_tools}")

    for tool in tools:
        if "jira" in tool.name.lower() or "confluence" in tool.name.lower():
            logger.info(f"Tool: {tool.name}")
            logger.info(f"  Description: {tool.description}")
            _log_schema(tool)

    interactive = os.getenv("VALIDATE_INTERACTIVE", "false").lower() == "true"
    tools = wrap_tools(tools, interactive=interactive)
    return tools


# ==========================================
# Agent main
# ==========================================
async def main(query: str):
    logger.info("Agent starting with query: %s", query)

    client = MultiServerMCPClient(server_params)
    try:
        tools = await validate_tools(client)
        agent = create_react_agent(model, tools)
        logger.info(f"Agent created with {len(tools)} tools")

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

        enhanced_query = f"{system_message}\n\nUser query: {query}"
        response = await agent.ainvoke({"messages": enhanced_query})
        logger.info("Agent received response.")
        logger.info(f"Response type: {type(response)}")
        if hasattr(response, "messages"):
            logger.info(f"Response messages: {len(response.messages)}")

    finally:
        pass

    return response


if __name__ == "__main__":
    # Examples
    # query = "Please get the in progress tickets from JIRA."
    # query = "please extract and tell me the roadmap of the project from confluence."
    query = "from the jira tickets, what should i do first ?"

    out = asyncio.run(main(query))
    print(out)
    print(last_ai_text(out))
