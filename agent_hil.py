# agent_hil_langgraph.py
import os
import json
import re
import asyncio
import logging
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, TypedDict

from load_dotenv import load_dotenv

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import StateGraph, END

# MCP client → LangChain tools
from langchain_mcp_adapters.client import MultiServerMCPClient


# =========================
# Logging
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

warnings.filterwarnings("ignore", category=ResourceWarning)
os.environ["PYTHONWARNINGS"] = "ignore::ResourceWarning"
load_dotenv()


# =========================
# Model
# =========================
# You can flip between Ollama and OpenAI. We'll keep both lines; the last one wins.
model = ChatOllama(model="qwen3:8b", temperature=0.1, streaming=True, base_url="http://localhost:11434")
model = ChatOpenAI(
    model="gpt-4.1",  # or "gpt-4o" etc.
    temperature=0.1,
    streaming=True,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


# =========================
# MCP servers
# =========================
raw_transport = os.environ.get("TRANSPORT", "streamable-http")
atlassian_transport = (
    "streamable_http"
    if raw_transport in {"streamable-http", "streamable_http"}
    else raw_transport
)
atlassian_default_path = "/mcp" if atlassian_transport == "streamable_http" else "/sse"
atlassian_default_url = (
    f"http://localhost:{os.environ.get('PORT', '10000')}{atlassian_default_path}"
)

server_params = {
    "atlassian": {
        "url": os.environ.get("ATLASSIAN_URL", atlassian_default_url),
        "transport": atlassian_transport,
    }
}


# =========================
# Normalization / Repair
# =========================
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
    # Rename legacy keys to server schema
    if "maxResults" in q and "limit" not in q:
        q["limit"] = q.pop("maxResults")
    if "start" in q and "start_at" not in q:
        q["start_at"] = q.pop("start")

    _clamp_limit(q)

    # Prefer double quotes inside JQL status
    if isinstance(q.get("jql"), str):
        q["jql"] = re.sub(r"status\s*=\s*'([^']+)'", r'status = "\1"', q["jql"])

        # Optional: inject env project filter
        proj_filter = os.getenv("JIRA_PROJECTS_FILTER", "").strip()
        if proj_filter and "project" not in q["jql"]:
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
    # Lightweight heuristics
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


# =========================
# HIL review prompt (console)
# =========================
async def hil_review(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Human-in-the-loop: approve/edit/reject tool calls.
    Returns edited params (dict) if approved, None if rejected.
    """
    pretty = json.dumps(params, indent=2, ensure_ascii=False)
    print(f"\n--- HIL Review ---\nTool: {tool_name}\nParams:\n{pretty}")
    try:
        choice = await asyncio.to_thread(input, "Approve? [y]/edit/n: ")
        choice = (choice or "y").strip().lower()
    except EOFError:
        choice = "y"
    if choice in {"y", ""}:
        return params
    if choice == "n":
        return None
    if choice == "edit":
        try:
            new_json = await asyncio.to_thread(input, "Enter JSON dict for new params: ")
            new_p = json.loads(new_json)
            if isinstance(new_p, dict):
                return new_p
            print("Not a JSON object; keeping original.")
            return params
        except Exception as e:
            print(f"Bad JSON ({e}); keeping original.")
            return params
    return params


# =========================
# Tool wrapping (structured; do not mutate originals)
# =========================
def wrap_tool(tool: BaseTool) -> BaseTool:
    """Return a new StructuredTool that proxies to the original with normalization + retry."""
    original_tool = tool
    tool_name = tool.name
    args_schema = getattr(tool, "args_schema", None)

    async def guarded_coroutine(**kwargs):
        params = kwargs if isinstance(kwargs, dict) else dict(kwargs)
        params = normalize_params(tool_name, params)

        # Validate against schema if exposed
        if args_schema:
            try:
                args_schema(**params)
            except Exception:
                # try normalize again; continue anyway
                params = normalize_params(tool_name, params)

        # HIL approval/edit
        edited = await hil_review(tool_name, params)
        if edited is None:
            raise RuntimeError(f"HIL rejected tool call: {tool_name}")

        logger.info(f"[TOOL CALL] {tool_name} -> {edited}")

        async def _call(p: Dict[str, Any]):
            if hasattr(original_tool, "ainvoke"):
                return await original_tool.ainvoke(p)
            return await asyncio.to_thread(original_tool.invoke, p)

        try:
            res = await _call(edited)
            logger.info(f"[TOOL OK] {tool_name}")
            return res
        except Exception as e:
            logger.warning(f"[TOOL ERR] {tool_name}: {e}")
            fix = auto_repair(tool_name, edited, e)
            if fix is not None:
                logger.info(f"[TOOL RETRY] {tool_name} -> {fix}")
                return await _call(fix)
            raise

    wrapped = StructuredTool.from_function(
        name=tool_name,
        description=tool.description,
        args_schema=args_schema,
        coroutine=guarded_coroutine,
    )
    return wrapped


def wrap_tools(tools: List[BaseTool]) -> List[BaseTool]:
    return [wrap_tool(t) for t in tools]


# =========================
# State & graph
# =========================
class AgentState(TypedDict):
    messages: List[BaseMessage]
    pending_tool: Optional[Dict[str, Any]]  # {"name": str, "args": dict, "id": str}
    steps: int


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
# Build the agent
# =========================
SYSTEM_PROMPT = """You are an AI assistant that helps users interact with JIRA and Confluence via MCP tools.

Use tools when needed. Prefer these parameter names:
- Jira search: use 'jql', 'limit', 'start_at'
- Confluence search: use 'query', 'limit'

Be concise. If you call a tool, do not explain the tool call; wait for the tool result and then produce the answer.
"""


async def load_tools() -> List[BaseTool]:
    client = MultiServerMCPClient(server_params)
    tools = await client.get_tools()
    # Optional logging
    logger.info(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")
    # Wrap tools with HIL guard
    tools = wrap_tools(tools)
    return tools


def build_graph(llm: ChatOpenAI | ChatOllama, tools: List[BaseTool]):
    tool_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(AgentState)

    async def node_llm(state: AgentState) -> AgentState:
        """Ask the model what to do next (tool call or final answer)."""
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        ai = await llm_with_tools.ainvoke(msgs)
        messages = state["messages"] + [ai]

        # Extract first tool call if any
        pending = None
        tool_calls = getattr(ai, "tool_calls", None)
        if tool_calls:
            tc = tool_calls[0]
            # tc has .name, .args, .id
            pending = {"name": tc["name"], "args": tc.get("args", {}), "id": tc.get("id")}

        return {"messages": messages, "pending_tool": pending, "steps": state["steps"] + 1}

    async def node_exec_tool(state: AgentState) -> AgentState:
        """Execute the pending tool (after HIL already approved inside wrapper)."""
        pending = state["pending_tool"]
        if not pending:
            return state

        name, args, tc_id = pending["name"], pending.get("args", {}), pending.get("id")
        tool = tool_by_name.get(name)
        if not tool:
            # No tool—append an explanation for the model
            warn = AIMessage(content=f"(Tool '{name}' not available. Please continue without it.)")
            return {"messages": state["messages"] + [warn], "pending_tool": None, "steps": state["steps"]}

        # Execute tool (our wrapper already did HIL + normalization)
        result = await tool.ainvoke(args)

        # Ensure plain string content to send back
        if not isinstance(result, str):
            try:
                content = json.dumps(result, ensure_ascii=False)
            except Exception:
                content = str(result)
        else:
            content = result

        tool_msg = ToolMessage(content=content, tool_call_id=tc_id or name)
        return {"messages": state["messages"] + [tool_msg], "pending_tool": None, "steps": state["steps"]}

    def route_after_llm(state: AgentState) -> str:
        # If the model asked for a tool, go exec; else we are done
        return "exec_tool" if state["pending_tool"] else END

    # Wire graph
    graph.add_node("llm", node_llm)
    graph.add_node("exec_tool", node_exec_tool)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", route_after_llm, {"exec_tool": "exec_tool", END: END})
    # After executing a tool, go back to llm to compose final answer (could request more tools)
    graph.add_edge("exec_tool", "llm")


    return graph.compile()


# =========================
# Runner
# =========================
async def run_agent(user_query: str):
    tools = await load_tools()
    app = build_graph(model, tools)
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")


    state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "pending_tool": None,
        "steps": 0,
    }

    # Run up to N steps to avoid infinite loops
    MAX_STEPS = 8
    for _ in range(MAX_STEPS):
        state = await app.ainvoke(state)
        # If last node ended (no pending tool and last message is AI, or we hit END), break
        # The compiled graph returns the updated state; when route_after_llm chooses END,
        # ainvoke returns with no further edges. We'll break if the last message is AI without tool calls.
        last = state["messages"][-1] if state["messages"] else None
        if isinstance(last, AIMessage):
            # If AI didn't ask for another tool (no pending), we can stop
            if not state["pending_tool"]:
                break

    # Print final assistant text
    final_text = None
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            final_text = m.content
            break

    return state, final_text


# =========================
# CLI
# =========================
if __name__ == "__main__":
    # Examples:
    # query = "please put some runescape music on Spotify and play them in the waiting list"
    # query = "Play my favorite songs on Spotify"
    # query = "Search for 'The Beatles' on Spotify"
    # query = "Please get the in progress tickets from JIRA."
    query = "Please tell me from confluence what the project is about."
    # query = "Get all the projects from Jira."
    # query = "can you comment the stories in progress a way to solve the stuff and a stuctured strategy to do it."
    # query = "please extract and tell me the roadmap of the project from confluence."
    # query ="from the jira tickets, what should i do first ?"
    # query ="Write a new page in confluence in the SHARP space and in the parent page called Project Management called 'project summary' Summarizing the project."
    # query ="give me the parent pages of the Sharp confluence space"
    # query ="Give me some pages in the confluence space 'sharp.'"
    query ="Give me some info on the product architecture and about the legal. Link the two and say how legal can be relevant for the architecture. "

    final_state, final_text = asyncio.run(run_agent(query))
    print("\n=== FINAL ANSWER ===\n")
    print(final_text or "<no AI text>")

