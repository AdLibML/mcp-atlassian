# agent_hil_langgraph.py
import os
import json
import re
import asyncio
import logging
import warnings
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
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
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
# You can flip between Ollama and OpenAI. The last assignment wins.
model = ChatOllama(model="qwen3:8b", temperature=0.1, streaming=True, base_url="http://localhost:11434")
model = ChatOpenAI(
    model="gpt-4.1",
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
def _clamp_limit(p: Dict[str, Any], *, key: str = "limit", min_v=1, max_v=50) -> Dict[str, Any]:
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


def _normalize_confluence_search(p: dict) -> dict:
    q = dict(p)
    if "cql" in q and "query" not in q:
        q["query"] = q.pop("cql")

    if isinstance(q.get("query"), str) and "type=space" in q["query"].replace(" ", ""):
        # CQL can't return spaces; switch to recent pages instead
        q["query"] = "type=page order by lastmodified desc"

    return _clamp_limit(q, key="limit", min_v=1, max_v=50)


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
HIL_AUTO_APPROVE = os.getenv("HIL_AUTO_APPROVE", "").strip().lower() in {"1", "true", "yes", "y"}


async def hil_review(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Human-in-the-loop: approve/edit/reject tool calls.
    Returns edited params (dict) if approved, None if rejected.
    """
    if HIL_AUTO_APPROVE:
        return params

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
    """Return a new StructuredTool that proxies to the original with normalization + HIL + auto-repair."""
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
            # Fallback in case the tool is sync-only
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
# Safe extraction of *multiple* tool calls
# =========================
def _safe_tool_calls(ai: AIMessage) -> List[Dict[str, Any]]:
    """
    Normalize tool calls from AIMessage to a list of dicts:
    [{'id': str, 'name': str, 'args': dict}]
    Works with OpenAI/LC shapes.
    """
    calls = getattr(ai, "tool_calls", None) or getattr(ai, "additional_kwargs", {}).get("tool_calls", [])
    out: List[Dict[str, Any]] = []
    for c in calls or []:
        # OpenAI style dict: {'id', 'type': 'function', 'function': {'name','arguments': json}}
        if isinstance(c, dict) and "function" in c:
            fn = c.get("function") or {}
            name = fn.get("name")
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {}
            out.append({"id": c.get("id"), "name": name, "args": args})
        else:
            # LangChain ToolCall shape
            name = getattr(c, "name", None) or (isinstance(c, dict) and c.get("name"))
            tid = getattr(c, "id", None) or (isinstance(c, dict) and c.get("id"))
            args = getattr(c, "args", None) or (isinstance(c, dict) and c.get("args")) or {}
            out.append({"id": tid, "name": name, "args": args})
    return out


# =========================
# State & helpers
# =========================
class AgentState(TypedDict):
    messages: List[BaseMessage]
    pending_tools: Optional[List[Dict[str, Any]]]  # list of {"id","name","args"}
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
# Build the agent (multi-tool safe)
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
    logger.info(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")
    tools = wrap_tools(tools)
    return tools


def build_graph(llm: ChatOpenAI | ChatOllama, tools: List[BaseTool]):
    tool_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(AgentState)

    async def node_llm(state: AgentState) -> AgentState:
        """Ask the model what to do next (possibly call multiple tools)."""
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        ai = await llm_with_tools.ainvoke(msgs)
        messages = state["messages"] + [ai]

        calls = _safe_tool_calls(ai)
        pending = calls if calls else None

        return {"messages": messages, "pending_tools": pending, "steps": state["steps"] + 1}

    async def node_exec_tool(state: AgentState) -> AgentState:
        """Execute all pending tools (each with HIL), return ToolMessages for each call id."""
        if not state.get("pending_tools"):
            return state

        tool_msgs: List[ToolMessage] = []
        for call in state["pending_tools"]:
            name = call.get("name")
            args = call.get("args", {}) or {}
            tc_id = call.get("id")
            tool = tool_by_name.get(name)

            if not tool:
                tool_msgs.append(
                    ToolMessage(
                        content=f"Error: unknown tool '{name}'",
                        name=name or "unknown_tool",
                        tool_call_id=tc_id or name or "unknown_call_id",
                    )
                )
                continue

            # Execute wrapped tool (does normalization + HIL + retry)
            result = await tool.ainvoke(args)

            # Ensure string content
            if isinstance(result, str):
                content = result
            else:
                try:
                    content = json.dumps(result, ensure_ascii=False)
                except Exception:
                    content = str(result)

            # IMPORTANT: include matching tool_call_id for EACH call
            tool_msgs.append(
                ToolMessage(
                    content=content,
                    name=name,
                    tool_call_id=tc_id or name,
                )
            )

        # Append all tool messages, clear pending; next step returns to LLM
        return {
            "messages": state["messages"] + tool_msgs,
            "pending_tools": None,
            "steps": state["steps"],
        }

    def route_after_llm(state: AgentState) -> str:
        # If the model asked for any tools, go exec; else we are done
        return "exec_tool" if state.get("pending_tools") else END

    # Wire graph
    graph.add_node("llm", node_llm)
    graph.add_node("exec_tool", node_exec_tool)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", route_after_llm, {"exec_tool": "exec_tool", END: END})
    # After executing tools, go back to llm to compose final answer (may ask for more tools)
    graph.add_edge("exec_tool", "llm")

    return graph.compile()


# =========================
# Runner
# =========================
async def run_agent(user_query: str):
    tools = await load_tools()
    app = build_graph(model, tools)

    # Optional: render the graph (requires graphviz/pygraphviz)
    try:
        app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    except Exception:
        pass

    state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "pending_tools": None,
        "steps": 0,
    }

    # Run up to N steps to avoid infinite loops
    MAX_STEPS = 8
    for _ in range(MAX_STEPS):
        state = await app.ainvoke(state)
        last = state["messages"][-1] if state["messages"] else None
        # If last was AI and there are no more pending tool calls, we're done
        if isinstance(last, AIMessage) and not state.get("pending_tools"):
            break

    # Pull final assistant text
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
    # query = "Please tell me from confluence what the project is about."
    # query = "Get all the projects from Jira."
    # query = "can you comment the stories in progress a way to solve the stuff and a stuctured strategy to do it."
    # query = "please extract and tell me the roadmap of the project from confluence."
    query = "from the jira tickets, what should i do first ?"
    # query ="Write a new page in confluence in the SHARP space and in the parent page called Project Management called 'project summary' Summarizing the project."
    # query ="give me the parent pages of the Sharp confluence space"
    # query ="Give me some pages in the confluence space 'sharp.'"
    # query ="Give me some info on the product architecture and about the legal. Link the two and say how legal can be relevant for the architecture. "
    # query = "Give all Confluence spaces you can see."
    # query = "From my personal space gary (space key  : ~712020d37b8394ef694a0dad694fb8e11026b4) give me the home content."


    final_state, final_text = asyncio.run(run_agent(query))
    print("\n=== FINAL ANSWER ===\n")
    print(final_text or "<no AI text>")
