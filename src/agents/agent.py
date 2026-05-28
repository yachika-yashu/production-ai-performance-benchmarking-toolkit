"""
LangGraph agent with full trajectory instrumentation.
Every reasoning step, tool call, and synthesis is individually measured.
"""

from __future__ import annotations

import time
from typing import Annotated, TypedDict
import operator

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.trajectory import AgentStep, AgentTrajectory, TrajectoryRecorder
from src.rag.corpus import get_document_by_id, get_all_documents

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
recorder = TrajectoryRecorder()

# ── Tools (replace with real tools in production) ─────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """Search the AI/ML knowledge base for information on a topic."""
    time.sleep(0.1)
    query_lower = query.lower()
    for doc in get_all_documents():
        keywords = doc["title"].lower().split() + doc["id"].split("-")
        if any(kw in query_lower for kw in keywords):
            return doc["content"][:600]
    return "No specific information found for that query."


@tool
def get_metric_definition(metric_name: str) -> str:
    """Look up the definition and target value for an AI evaluation metric."""
    time.sleep(0.05)
    metrics = {
        "faithfulness": "Faithfulness measures whether the answer is grounded in context. Target: ≥0.85.",
        "context_recall": "Context recall measures whether retrieved context covers the answer. Target: ≥0.80.",
        "answer_relevancy": "Answer relevancy measures whether the answer addresses the question. Target: ≥0.80.",
        "context_precision": "Context precision measures retrieval noise. Target: ≥0.75.",
        "p99": "P99 latency is the 99th percentile response time — worst-case user experience.",
        "ttft": "Time to First Token — latency until streaming begins. Key for interactive apps.",
    }
    name = metric_name.lower().replace(" ", "_")
    return metrics.get(name, f"Definition not found for '{metric_name}'.")


@tool
def compare_approaches(approach_a: str, approach_b: str) -> str:
    """Compare two AI/ML approaches or techniques."""
    time.sleep(0.1)
    return (
        f"Comparison: {approach_a} vs {approach_b}\n"
        f"- {approach_a}: Established approach with broad applicability.\n"
        f"- {approach_b}: Alternative with different tradeoffs.\n"
        f"Key consideration: depends on your specific constraints (latency, cost, quality)."
    )


# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    query: str
    trajectory: AgentTrajectory
    step_count: int


# ── Graph nodes ───────────────────────────────────────────────────────────────

_tools = [search_knowledge_base, get_metric_definition, compare_approaches]
_llm_with_tools = _llm.bind_tools(_tools)
_tool_executor = {t.name: t for t in _tools}


def reasoning_node(state: AgentState) -> AgentState:
    start = time.perf_counter()
    response = _llm_with_tools.invoke(state["messages"])
    latency_ms = (time.perf_counter() - start) * 1000

    step = AgentStep(
        step_number=state["step_count"] + 1,
        step_type="reasoning",
        reasoning=response.content[:200] if response.content else None,
        latency_ms=latency_ms,
    )
    state["trajectory"].add_step(step)
    return {**state, "messages": [response], "step_count": state["step_count"] + 1}


def tool_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    tool_calls = last.tool_calls if hasattr(last, "tool_calls") else []
    new_messages = []

    for call in tool_calls:
        start = time.perf_counter()
        try:
            fn = _tool_executor.get(call["name"])
            result = fn.invoke(call["args"]) if fn else f"Unknown tool: {call['name']}"
            success, error = True, None
        except Exception as e:
            result, success, error = str(e), False, str(e)

        latency_ms = (time.perf_counter() - start) * 1000
        state["trajectory"].add_step(AgentStep(
            step_number=state["step_count"] + 1,
            step_type="tool_call",
            tool_name=call["name"],
            tool_input=call["args"],
            tool_output=str(result)[:200],
            latency_ms=latency_ms,
            success=success,
            error=error,
        ))
        new_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return {**state, "messages": new_messages, "step_count": state["step_count"] + 1}


def synthesis_node(state: AgentState) -> AgentState:
    start = time.perf_counter()
    prompt = HumanMessage(content="Provide a clear final answer to the original question based on all gathered information.")
    response = _llm.invoke(state["messages"] + [prompt])
    latency_ms = (time.perf_counter() - start) * 1000

    state["trajectory"].add_step(AgentStep(
        step_number=state["step_count"] + 1,
        step_type="synthesis",
        reasoning=response.content[:200],
        latency_ms=latency_ms,
    ))
    state["trajectory"].final_answer = response.content
    state["trajectory"].compute_metrics(min_steps_expected=2)
    return {**state, "messages": [response], "step_count": state["step_count"] + 1}


def should_continue(state: AgentState) -> str:
    MAX_STEPS = 8
    if state["step_count"] >= MAX_STEPS:
        state["trajectory"].loop_detected = True
        return "synthesize"
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "use_tool"
    return "synthesize"


# ── Build graph ───────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("reason", reasoning_node)
    g.add_node("use_tool", tool_node)
    g.add_node("synthesize", synthesis_node)
    g.set_entry_point("reason")
    g.add_conditional_edges("reason", should_continue, {"use_tool": "use_tool", "synthesize": "synthesize"})
    g.add_edge("use_tool", "reason")
    g.add_edge("synthesize", END)
    return g.compile()


def run_agent(query: str) -> AgentTrajectory:
    graph = _build_graph()
    trajectory = recorder.new_trajectory(query)
    try:
        result = graph.invoke({
            "messages": [HumanMessage(content=query)],
            "query": query,
            "trajectory": trajectory,
            "step_count": 0,
        })
        trajectory = result["trajectory"]
        trajectory.success = True
    except Exception as e:
        trajectory.success = False
        trajectory.final_answer = f"Agent failed: {e}"
        trajectory.compute_metrics()
    recorder.record(trajectory)
    return trajectory
