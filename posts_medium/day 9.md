Here we go.

---

# AI Performance Benchmarking: From Zero to Production
### Day 9 of 12 — Agentic Benchmarking: When Your AI Takes Multiple Steps To Answer

---

*This is Day 9 of a 12-part series. Start with **[Day 0](#)** if you're just joining.*

---

## The Benchmark That Passed. The Agent That Failed.

Everything looked great on paper.

Faithfulness: 0.91.
Latency: 1.2 seconds.
Cost: within budget.

You built an agent.

Not a simple RAG pipeline — a proper multi-step agent. It searches documents, calls external tools, reasons across multiple sources, and synthesizes a final answer.

You ran your Day 4 quality metrics on it.

They passed.

Then someone used it in production.

The agent called the wrong tool three times before finding the right one.
It retrieved the same document twice and charged you for it.
On one query it got stuck in a reasoning loop — calling itself in circles — until it hit the token limit and returned nothing.

Your RAGAS scores never caught any of this.

Because RAGAS measures the final answer.

**Agents do not just produce answers. They produce trajectories.**

And trajectories need their own benchmarks.

---

## What Makes Agentic Benchmarking Different

A simple RAG pipeline has one path:

```
Query → Retrieve → Generate → Answer
```

An agent has many possible paths:

```
Query → Think → Choose Tool → Call Tool → Observe Result
      → Think → Choose Tool → Call Tool → Observe Result
      → Think → Synthesize → Answer
```

Or sometimes:

```
Query → Think → Wrong Tool → Error
      → Think → Wrong Tool → Error
      → Think → Right Tool → Answer (eventually)
```

Or worst case:

```
Query → Think → Tool A → Think → Tool A → Think → Tool A
      → ... → Token limit → Empty response
```

Every one of those paths is a different failure mode.

And none of them are visible in a final-answer quality score.

---

## The Five Agentic Failure Modes

Before we benchmark anything — know what you are looking for.

---

**Failure Mode 1 — Wrong tool selection**

The agent has access to multiple tools. It picks the wrong one for the query.

Not an error. The tool runs fine. But the output is irrelevant to what was asked.

The agent then tries to build an answer from irrelevant tool output.

Result: confident wrong answer. High faithfulness. Wrong information.

---

**Failure Mode 2 — Reasoning loops**

The agent calls the same tool repeatedly with slight variations, never satisfied with the result.

Each iteration costs tokens and time.

Eventually hits the maximum iteration limit or token limit and either returns an empty response or a partial one.

---

**Failure Mode 3 — Unnecessary tool calls**

The agent calls three tools when one would have sufficed.

The answer was available after step one. The agent did not recognize it.

Result: 3x the cost. 3x the latency. Same quality as a simpler pipeline.

---

**Failure Mode 4 — Step-level quality degradation**

Each individual step looks fine.

But errors compound. A slightly wrong retrieval in step 1 leads to a slightly wrong reasoning in step 2, which leads to a confidently wrong synthesis in step 3.

End-to-end quality metrics hide this because they only see the final output.

---

**Failure Mode 5 — Tool call accuracy**

The agent formats tool calls incorrectly — wrong parameters, wrong types, missing required fields.

The tool returns an error. The agent tries to handle the error. Sometimes it succeeds. Sometimes it spirals.

---

## The Metrics That Actually Matter For Agents

---

**Trajectory Length**

How many steps did the agent take to answer the query?

A query that should take 2 steps but took 7 is a signal. Either the agent is inefficient, looping, or genuinely struggling with complexity.

---

**Tool Call Accuracy**

What fraction of tool calls were correctly formatted and returned a valid result?

Low tool call accuracy means your agent is wasting steps recovering from errors it should not be making.

---

**Step-Level Quality**

Was the output of each individual step correct?

Not just the final answer — each intermediate result.

A step that retrieves the wrong document contaminates every step that follows.

---

**Trajectory Efficiency**

Did the agent take the minimum number of steps necessary?

Efficiency = minimum possible steps / actual steps taken.

1.0 is perfect. 0.3 means the agent took 3x more steps than needed.

---

**Final Answer Quality**

The RAGAS scores from Day 4. Still relevant — but now contextualized against trajectory quality.

A correct answer via an inefficient trajectory is a different problem than a correct answer via an efficient one.

---

## LangGraph — Why It Makes Benchmarking Easier

LangGraph structures agents as explicit graphs — nodes are steps, edges are transitions.

This explicit structure makes benchmarking natural:

- Every node execution is a measurable step
- Every edge traversal is a trackable transition
- Loops are visible as cycles in the graph
- Dead ends are visible as nodes with no exit

We use LangGraph today so every agent step is individually measurable.

---

## The Problem Most People Hit

You want to benchmark your agent.

So you wrap the whole thing in a timer and check the final answer quality.

You are measuring a black box.

You know the input went in and something came out.

Everything that happened in between — every tool call, every reasoning step, every intermediate result — is invisible.

Today we open the black box.

---

## The Code — Agentic Benchmarking Suite

Today we build four things:

1. A **LangGraph agent** with instrumented steps
2. A **trajectory recorder** that captures every step
3. An **agentic metrics calculator** — trajectory length, tool accuracy, efficiency
4. A **step-level quality evaluator** that checks intermediate results

---

### Step 1 — Set up today's folder

```bash
cd ..
mkdir day-09-agentic
cd day-09-agentic
```

---

### Step 2 — Install dependencies

```bash
pip install langchain langchain-openai langgraph python-dotenv ragas datasets pandas numpy
```

⚠️ **If langgraph install fails:**
```bash
pip install langgraph==0.1.40
```

---

### Step 3 — Create the trajectory recorder

Right-click `day-09-agentic` → **New File** → name it:

```
trajectory_recorder.py
```

Paste this exactly:

```python
# trajectory_recorder.py
# Records every step an agent takes
# This is the foundation of agentic benchmarking

from dotenv import load_dotenv
load_dotenv()

import time
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


# ─────────────────────────────────────────────
# STEP RECORD — one agent step
# ─────────────────────────────────────────────

@dataclass
class AgentStep:
    step_number: int
    step_type: str          # "reasoning", "tool_call", "synthesis"
    tool_name: Optional[str] = None
    tool_input: Optional[Dict] = None
    tool_output: Optional[str] = None
    reasoning: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def summary(self) -> str:
        status = "✓" if self.success else "✗"
        tool_info = f" → {self.tool_name}" if self.tool_name else ""
        error_info = f" [ERROR: {self.error}]" if self.error else ""
        return (
            f"  Step {self.step_number}: "
            f"{status} {self.step_type}{tool_info} "
            f"({self.latency_ms:.0f}ms)"
            f"{error_info}"
        )


# ─────────────────────────────────────────────
# TRAJECTORY — complete agent run
# ─────────────────────────────────────────────

@dataclass
class AgentTrajectory:
    query: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    success: bool = True

    # Computed metrics
    trajectory_length: int = 0
    tool_call_accuracy: float = 0.0
    trajectory_efficiency: float = 0.0
    loop_detected: bool = False
    timestamp: str = ""

    def __post_init__(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def add_step(self, step: AgentStep):
        self.steps.append(step)
        self.total_latency_ms += step.latency_ms
        self.total_tokens += step.tokens_used

    def compute_metrics(self, min_steps_expected: int = 2):
        """
        Computes all trajectory metrics.
        Call this after the agent finishes.

        min_steps_expected: the minimum steps a correct
        agent should take for this type of query.
        Use 2 for simple queries, 3-4 for complex ones.
        """
        self.trajectory_length = len(self.steps)

        # Tool call accuracy
        tool_steps = [s for s in self.steps if s.step_type == "tool_call"]
        if tool_steps:
            successful_tools = sum(1 for s in tool_steps if s.success)
            self.tool_call_accuracy = successful_tools / len(tool_steps)
        else:
            self.tool_call_accuracy = 1.0

        # Trajectory efficiency
        if self.trajectory_length > 0:
            self.trajectory_efficiency = min(
                min_steps_expected / self.trajectory_length, 1.0
            )

        # Loop detection
        # A loop is when the same tool is called 3+ times with similar inputs
        tool_calls = [
            s.tool_name for s in self.steps
            if s.step_type == "tool_call" and s.tool_name
        ]
        if tool_calls:
            from collections import Counter
            tool_counts = Counter(tool_calls)
            self.loop_detected = any(
                count >= 3 for count in tool_counts.values()
            )

    def summary(self) -> str:
        loop_warning = " ⚠️  LOOP DETECTED" if self.loop_detected else ""

        lines = [
            f"\n{'━'*60}",
            f"🤖 AGENT TRAJECTORY",
            f"{'━'*60}",
            f"Query:      {self.query[:55]}"
            f"{'...' if len(self.query) > 55 else ''}",
            f"Steps:      {self.trajectory_length}{loop_warning}",
            f"Efficiency: {self.trajectory_efficiency:.2f} "
            f"(1.0 = optimal)",
            f"Tool Acc:   {self.tool_call_accuracy:.2f}",
            f"Latency:    {self.total_latency_ms:.0f}ms",
            f"Tokens:     {self.total_tokens}",
            f"{'─'*60}",
            f"Steps taken:"
        ]

        for step in self.steps:
            lines.append(step.summary())

        lines.extend([
            f"{'─'*60}",
            f"Answer:     {self.final_answer[:80]}"
            f"{'...' if len(self.final_answer) > 80 else ''}",
            f"{'━'*60}"
        ])

        return "\n".join(lines)


# ─────────────────────────────────────────────
# TRAJECTORY RECORDER
# Collects and analyzes multiple trajectories
# ─────────────────────────────────────────────

class TrajectoryRecorder:

    def __init__(self):
        self.trajectories: List[AgentTrajectory] = []

    def new_trajectory(self, query: str) -> AgentTrajectory:
        """Start recording a new trajectory"""
        trajectory = AgentTrajectory(query=query)
        return trajectory

    def record(self, trajectory: AgentTrajectory):
        """Save a completed trajectory"""
        self.trajectories.append(trajectory)

    def print_aggregate_report(self):
        """Aggregate metrics across all recorded trajectories"""

        if not self.trajectories:
            print("No trajectories recorded yet.")
            return

        lengths = [t.trajectory_length for t in self.trajectories]
        efficiencies = [t.trajectory_efficiency for t in self.trajectories]
        tool_accs = [t.tool_call_accuracy for t in self.trajectories]
        latencies = [t.total_latency_ms for t in self.trajectories]
        loops = sum(1 for t in self.trajectories if t.loop_detected)
        failures = sum(1 for t in self.trajectories if not t.success)

        print(f"\n{'━'*60}")
        print(f"📊 AGENTIC BENCHMARK REPORT")
        print(f"   {len(self.trajectories)} trajectories analyzed")
        print(f"{'━'*60}")
        print(f"  Avg trajectory length:   {np.mean(lengths):.1f} steps")
        print(f"  Max trajectory length:   {max(lengths)} steps")
        print(f"  Avg efficiency:          {np.mean(efficiencies):.2f}")
        print(f"  Avg tool call accuracy:  {np.mean(tool_accs):.2f}")
        print(f"  Avg latency:             {np.mean(latencies):.0f}ms")
        print(f"  Loops detected:          {loops}/{len(self.trajectories)}")
        print(f"  Agent failures:          {failures}/{len(self.trajectories)}")
        print(f"{'─'*60}")

        # Health assessment
        avg_eff = np.mean(efficiencies)
        avg_acc = np.mean(tool_accs)

        if avg_eff >= 0.8 and avg_acc >= 0.9 and loops == 0:
            print(f"  Status: ✅ Agent is performing well")
        elif avg_eff >= 0.6 and avg_acc >= 0.75:
            print(f"  Status: 🟡 Agent needs optimization")
            if avg_eff < 0.6:
                print(f"    → Too many steps per query — review tool design")
            if avg_acc < 0.75:
                print(f"    → Tool call errors — review tool schemas")
        else:
            print(f"  Status: 🔴 Agent has significant issues")
            if loops > 0:
                print(f"    → Loops detected — add loop prevention")
            if failures > 0:
                print(f"    → Agent failures — check error handling")

        print(f"{'━'*60}\n")

    def save(self, filename: str = "trajectories.json"):
        data = []
        for t in self.trajectories:
            data.append({
                "query": t.query,
                "trajectory_length": t.trajectory_length,
                "tool_call_accuracy": t.tool_call_accuracy,
                "trajectory_efficiency": t.trajectory_efficiency,
                "loop_detected": t.loop_detected,
                "total_latency_ms": t.total_latency_ms,
                "total_tokens": t.total_tokens,
                "success": t.success,
                "steps": [
                    {
                        "step_number": s.step_number,
                        "step_type": s.step_type,
                        "tool_name": s.tool_name,
                        "latency_ms": s.latency_ms,
                        "success": s.success,
                        "error": s.error
                    }
                    for s in t.steps
                ],
                "final_answer": t.final_answer[:200],
                "timestamp": t.timestamp
            })

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Trajectories saved → {filename}")
```

---

### Step 4 — Create the LangGraph agent

Right-click `day-09-agentic` → **New File** → name it:

```
benchmarked_agent.py
```

Paste this:

```python
# benchmarked_agent.py
# A LangGraph agent with full trajectory instrumentation
# Every step is measured and recorded
# Replace tools with your real tools

from dotenv import load_dotenv
load_dotenv()

import time
import json
from typing import TypedDict, Annotated, List
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langsmith import traceable

from trajectory_recorder import (
    TrajectoryRecorder,
    AgentTrajectory,
    AgentStep
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
recorder = TrajectoryRecorder()


# ─────────────────────────────────────────────
# DEFINE YOUR TOOLS
# Replace these with your real tools
# Each tool is one node in the agent graph
# ─────────────────────────────────────────────

@tool
def search_documents(query: str) -> str:
    """
    Search the document store for relevant information.
    Use this to find facts, definitions, and explanations.
    """
    # Replace with your real retriever
    time.sleep(0.15)  # Simulates retrieval latency

    # Simulated document search results
    knowledge_base = {
        "kafka": "Apache Kafka is a distributed event streaming platform "
                 "developed at LinkedIn. It uses topics, partitions, and "
                 "brokers for scalable message processing.",
        "partition": "A Kafka partition is an ordered sequence of messages. "
                     "Topics are divided into partitions for parallel "
                     "processing and scalability.",
        "consumer": "Kafka consumers read messages from topics. Consumer "
                    "groups allow parallel processing by assigning partitions "
                    "to individual consumers.",
        "broker": "A Kafka broker is a server that stores and serves data. "
                  "A cluster consists of multiple brokers for fault tolerance.",
        "offset": "A Kafka offset is a unique integer identifying a message's "
                  "position within a partition. Consumers track offsets to "
                  "know where they left off.",
        "replication": "Kafka replicates partitions across multiple brokers. "
                       "Each partition has one leader and multiple followers "
                       "for fault tolerance."
    }

    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value

    return (
        "No specific information found. "
        "Try a more specific search term."
    )


@tool
def calculate_metrics(
    metric_type: str,
    values: List[float]
) -> str:
    """
    Calculate performance metrics from a list of values.
    metric_type: 'average', 'p99', 'max', 'min'
    values: list of numeric values to analyze
    """
    # Replace with your real metrics calculator
    import numpy as np

    if not values:
        return "No values provided"

    if metric_type == "average":
        return f"Average: {np.mean(values):.2f}"
    elif metric_type == "p99":
        return f"P99: {np.percentile(values, 99):.2f}"
    elif metric_type == "max":
        return f"Maximum: {max(values):.2f}"
    elif metric_type == "min":
        return f"Minimum: {min(values):.2f}"
    else:
        return f"Unknown metric type: {metric_type}"


@tool
def compare_options(
    option_a: str,
    option_b: str,
    criteria: str
) -> str:
    """
    Compare two options based on specified criteria.
    Returns a structured comparison.
    """
    # Replace with your real comparison logic
    time.sleep(0.1)
    return (
        f"Comparison of {option_a} vs {option_b} "
        f"on {criteria}:\n"
        f"- {option_a}: Established option with broad ecosystem\n"
        f"- {option_b}: Alternative with different tradeoffs\n"
        f"- Key difference: depends on {criteria} requirements"
    )


# ─────────────────────────────────────────────
# AGENT STATE
# Everything the agent knows at each step
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    query: str
    trajectory: AgentTrajectory
    step_count: int


# ─────────────────────────────────────────────
# AGENT NODES
# Each node is one step in the graph
# ─────────────────────────────────────────────

tools = [search_documents, calculate_metrics, compare_options]
llm_with_tools = llm.bind_tools(tools)
tool_executor = {t.name: t for t in tools}


@traceable
def reasoning_node(state: AgentState) -> AgentState:
    """
    The agent thinks about what to do next.
    Decides whether to use a tool or synthesize a final answer.
    """
    start = time.perf_counter()

    response = llm_with_tools.invoke(state["messages"])

    latency_ms = (time.perf_counter() - start) * 1000

    # Record this step
    step = AgentStep(
        step_number=state["step_count"] + 1,
        step_type="reasoning",
        reasoning=response.content[:200] if response.content else None,
        latency_ms=latency_ms,
        tokens_used=response.usage_metadata.get(
            "total_tokens", 0
        ) if hasattr(response, 'usage_metadata') and \
            response.usage_metadata else 0
    )
    state["trajectory"].add_step(step)

    return {
        **state,
        "messages": [response],
        "step_count": state["step_count"] + 1
    }


@traceable
def tool_node(state: AgentState) -> AgentState:
    """
    Executes the tool the agent chose.
    Records success, failure, and latency per tool call.
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls if hasattr(
        last_message, 'tool_calls'
    ) else []

    new_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]

        start = time.perf_counter()

        try:
            # Execute the tool
            tool_fn = tool_executor.get(tool_name)
            if tool_fn:
                result = tool_fn.invoke(tool_input)
                success = True
                error = None
            else:
                result = f"Tool '{tool_name}' not found"
                success = False
                error = f"Unknown tool: {tool_name}"

        except Exception as e:
            result = f"Tool error: {str(e)}"
            success = False
            error = str(e)

        latency_ms = (time.perf_counter() - start) * 1000

        # Record this tool call
        step = AgentStep(
            step_number=state["step_count"] + 1,
            step_type="tool_call",
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=str(result)[:200],
            latency_ms=latency_ms,
            success=success,
            error=error
        )
        state["trajectory"].add_step(step)

        new_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    return {
        **state,
        "messages": new_messages,
        "step_count": state["step_count"] + 1
    }


def synthesis_node(state: AgentState) -> AgentState:
    """
    Final step — synthesizes the answer from all gathered information.
    """
    start = time.perf_counter()

    # Add explicit synthesis instruction
    synthesis_prompt = HumanMessage(
        content="Based on all the information gathered, "
                "provide a clear and complete final answer "
                "to the original question."
    )

    response = llm.invoke(state["messages"] + [synthesis_prompt])
    latency_ms = (time.perf_counter() - start) * 1000

    step = AgentStep(
        step_number=state["step_count"] + 1,
        step_type="synthesis",
        reasoning=response.content[:200],
        latency_ms=latency_ms
    )
    state["trajectory"].add_step(step)

    # Finalize trajectory
    state["trajectory"].final_answer = response.content
    state["trajectory"].compute_metrics(min_steps_expected=2)

    return {
        **state,
        "messages": [response],
        "step_count": state["step_count"] + 1
    }


# ─────────────────────────────────────────────
# ROUTING LOGIC
# Decides what happens after each reasoning step
# ─────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """
    After reasoning — should we call a tool or synthesize?

    Also implements loop prevention:
    If the agent has taken too many steps, force synthesis.
    """
    # Loop prevention — never exceed max steps
    MAX_STEPS = 8
    if state["step_count"] >= MAX_STEPS:
        state["trajectory"].loop_detected = True
        return "synthesize"

    last_message = state["messages"][-1]

    # If the model wants to use a tool — go to tool node
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "use_tool"

    # Otherwise synthesize
    return "synthesize"


# ─────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_agent():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("reason", reasoning_node)
    graph.add_node("use_tool", tool_node)
    graph.add_node("synthesize", synthesis_node)

    # Set entry point
    graph.set_entry_point("reason")

    # Conditional routing after reasoning
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {
            "use_tool": "use_tool",
            "synthesize": "synthesize"
        }
    )

    # After tool use — always reason again
    graph.add_edge("use_tool", "reason")

    # After synthesis — done
    graph.add_edge("synthesize", END)

    return graph.compile()


# ─────────────────────────────────────────────
# RUN THE AGENT
# ─────────────────────────────────────────────

@traceable
def run_agent(query: str) -> AgentTrajectory:
    """
    Runs the agent on a query and returns the full trajectory.
    This is what you benchmark.
    """
    agent = build_agent()

    trajectory = recorder.new_trajectory(query)

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "trajectory": trajectory,
        "step_count": 0
    }

    try:
        result = agent.invoke(initial_state)
        trajectory = result["trajectory"]
        trajectory.success = True
    except Exception as e:
        trajectory.success = False
        trajectory.final_answer = f"Agent failed: {str(e)}"
        trajectory.compute_metrics()

    recorder.record(trajectory)
    return trajectory
```

---

### Step 5 — Create the step-level quality evaluator

Right-click `day-09-agentic` → **New File** → name it:

```
step_quality_evaluator.py
```

Paste this:

```python
# step_quality_evaluator.py
# Evaluates quality at each agent step — not just the final answer
# Catches error propagation before it reaches the final output

from dotenv import load_dotenv
load_dotenv()

import json
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import traceable
from trajectory_recorder import AgentTrajectory


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


@traceable
def evaluate_step_quality(
    query: str,
    step_output: str,
    step_type: str,
    expected_direction: str
) -> Dict:
    """
    Evaluates whether a single agent step output is
    moving toward a correct final answer.

    Parameters:
    - query:               Original user question
    - step_output:         What this step produced
    - step_type:           "tool_call" or "reasoning"
    - expected_direction:  What a correct step should produce

    Returns quality assessment for this step.
    """

    prompt = f"""You are evaluating one step of an AI agent's reasoning.

Original question: {query}

Step type: {step_type}
Step output: {step_output}
Expected direction: {expected_direction}

Evaluate this step on three dimensions:
1. Relevance (0-1): Is this step output relevant to answering the question?
2. Correctness (0-1): Is the information accurate and useful?
3. Progress (0-1): Does this step move toward the correct answer?

Respond in JSON only:
{{
  "relevance": 0.0,
  "correctness": 0.0,
  "progress": 0.0,
  "assessment": "brief explanation",
  "red_flag": true/false
}}

red_flag should be true if this step contains wrong information
that could corrupt the final answer."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        # Parse JSON response
        content = response.content
        start = content.find('{')
        end = content.rfind('}') + 1
        result = json.loads(content[start:end])
        result["step_type"] = step_type
        return result
    except Exception:
        return {
            "relevance": 0.5,
            "correctness": 0.5,
            "progress": 0.5,
            "assessment": "Could not parse evaluation",
            "red_flag": False,
            "step_type": step_type
        }


def evaluate_trajectory_quality(
    trajectory: AgentTrajectory,
    expected_directions: List[str] = None
) -> Dict:
    """
    Evaluates quality at every step of a trajectory.
    Shows where errors enter and how they propagate.

    expected_directions: what each step should ideally produce.
    If not provided uses generic expectations.
    """

    tool_steps = [
        s for s in trajectory.steps
        if s.step_type == "tool_call" and s.tool_output
    ]

    if not tool_steps:
        return {
            "message": "No tool call steps to evaluate",
            "step_scores": []
        }

    step_scores = []
    red_flags = 0

    print(f"\n  Evaluating {len(tool_steps)} tool call steps...")

    for i, step in enumerate(tool_steps):
        expected = (
            expected_directions[i]
            if expected_directions and i < len(expected_directions)
            else "Retrieve accurate and relevant information"
        )

        score = evaluate_step_quality(
            query=trajectory.query,
            step_output=step.tool_output or "",
            step_type=step.step_type,
            expected_direction=expected
        )

        score["step_number"] = step.step_number
        score["tool_name"] = step.tool_name
        step_scores.append(score)

        flag = "🚩" if score.get("red_flag") else "  "
        print(
            f"    {flag} Step {step.step_number} "
            f"({step.tool_name}): "
            f"relevance={score['relevance']:.2f}  "
            f"correctness={score['correctness']:.2f}  "
            f"progress={score['progress']:.2f}"
        )

        if score.get("red_flag"):
            red_flags += 1
            print(
                f"       ⚠️  Red flag: {score.get('assessment', '')}"
            )

    avg_relevance = sum(
        s["relevance"] for s in step_scores
    ) / len(step_scores)
    avg_correctness = sum(
        s["correctness"] for s in step_scores
    ) / len(step_scores)
    avg_progress = sum(
        s["progress"] for s in step_scores
    ) / len(step_scores)

    return {
        "step_scores": step_scores,
        "avg_relevance": round(avg_relevance, 3),
        "avg_correctness": round(avg_correctness, 3),
        "avg_progress": round(avg_progress, 3),
        "red_flags": red_flags,
        "step_count": len(tool_steps)
    }
```

---

### Step 6 — Create the runner

Right-click `day-09-agentic` → **New File** → name it:

```
run_agentic_benchmark.py
```

Paste this:

```python
# run_agentic_benchmark.py
# Runs the complete agentic benchmarking suite

from dotenv import load_dotenv
load_dotenv()

from benchmarked_agent import run_agent, recorder
from step_quality_evaluator import evaluate_trajectory_quality

# ─────────────────────────────────────────────
# TEST QUERIES
# Mix of simple, moderate, and complex
# Replace with your real agent queries
# ─────────────────────────────────────────────

TEST_QUERIES = [
    # Simple — should take 2 steps (1 tool + synthesis)
    {
        "query": "What is Apache Kafka?",
        "min_steps_expected": 2,
        "description": "simple_factual"
    },
    # Moderate — should take 3 steps (2 tools + synthesis)
    {
        "query": "How do Kafka partitions and consumer groups "
                 "work together for parallel processing?",
        "min_steps_expected": 3,
        "description": "moderate_multi_concept"
    },
    # Complex — should take 4 steps (3 tools + synthesis)
    {
        "query": "Compare Kafka brokers and partitions, then "
                 "calculate what the average replication factor "
                 "would be if we have 3 brokers and 9 partitions.",
        "min_steps_expected": 4,
        "description": "complex_multi_tool"
    },
    # Edge case — agent should recognize it needs minimal steps
    {
        "query": "What is an offset?",
        "min_steps_expected": 2,
        "description": "simple_definition"
    }
]


if __name__ == "__main__":

    print("=" * 60)
    print("AGENTIC BENCHMARKING SUITE")
    print("=" * 60)

    all_step_quality = []

    for test in TEST_QUERIES:
        print(f"\n{'─'*60}")
        print(f"Query: {test['query'][:55]}...")
        print(f"Type:  {test['description']}")
        print(f"{'─'*60}")

        # Run the agent
        trajectory = run_agent(test["query"])

        # Override min_steps for efficiency calculation
        trajectory.compute_metrics(
            min_steps_expected=test["min_steps_expected"]
        )

        # Print trajectory summary
        print(trajectory.summary())

        # Evaluate step-level quality
        print(f"\n  Step-level quality evaluation:")
        step_quality = evaluate_trajectory_quality(trajectory)
        all_step_quality.append(step_quality)

        # Print red flag warning
        if step_quality.get("red_flags", 0) > 0:
            print(
                f"\n  🚩 {step_quality['red_flags']} red flag(s) detected"
                f" — errors may have propagated to final answer"
            )

    # Print aggregate report
    recorder.print_aggregate_report()

    # Print step quality summary
    if all_step_quality:
        valid = [
            s for s in all_step_quality
            if s.get("step_count", 0) > 0
        ]
        if valid:
            avg_correctness = sum(
                s["avg_correctness"] for s in valid
            ) / len(valid)
            total_red_flags = sum(
                s.get("red_flags", 0) for s in all_step_quality
            )

            print(f"\n{'━'*60}")
            print(f"📊 STEP QUALITY SUMMARY")
            print(f"{'━'*60}")
            print(
                f"  Avg step correctness:  {avg_correctness:.3f}"
            )
            print(
                f"  Total red flags:       {total_red_flags}"
            )
            if total_red_flags > 0:
                print(
                    f"  ⚠️  Review flagged steps — "
                    f"these are contaminating final answers"
                )
            print(f"{'━'*60}")

    # Save everything
    recorder.save("trajectories.json")

    print(
        "\n✅ Check LangSmith — all agent runs logged "
        "with full step breakdown."
    )
```

Run it:

```bash
python run_agentic_benchmark.py
```

You should see:

```
============================================================
AGENTIC BENCHMARKING SUITE
============================================================

────────────────────────────────────────────────────────────
Query: What is Apache Kafka?
Type:  simple_factual
────────────────────────────────────────────────────────────

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 AGENT TRAJECTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Query:      What is Apache Kafka?
Steps:      3
Efficiency: 0.67 (1.0 = optimal)
Tool Acc:   1.00
Latency:    2341ms
Tokens:     847
────────────────────────────────────────────────────────────
Steps taken:
  Step 1: ✓ reasoning (412ms)
  Step 2: ✓ tool_call → search_documents (187ms)
  Step 3: ✓ synthesis (891ms)
────────────────────────────────────────────────────────────
Answer:     Apache Kafka is a distributed event streaming platform...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Step-level quality evaluation:
    Step 2 (search_documents): relevance=0.94  correctness=0.91  progress=0.93

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 AGENTIC BENCHMARK REPORT
   4 trajectories analyzed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Avg trajectory length:   3.2 steps
  Max trajectory length:   5 steps
  Avg efficiency:          0.71
  Avg tool call accuracy:  0.94
  Avg latency:             3821ms
  Loops detected:          0/4
  Agent failures:          0/4
────────────────────────────────────────────────────────────
  Status: 🟡 Agent needs optimization
    → Too many steps per query — review tool design
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Every step visible. Every failure mode detectable. Nothing hidden.

---

### Step 7 — Plug into your existing agent

Three changes for your real LangGraph agent:

**Change 1 — Import the recorder:**

```python
from trajectory_recorder import TrajectoryRecorder, AgentStep
recorder = TrajectoryRecorder()
```

**Change 2 — Add step recording to your existing nodes:**

```python
# In each of your existing node functions add:
step = AgentStep(
    step_number=state["step_count"] + 1,
    step_type="tool_call",     # or "reasoning" or "synthesis"
    tool_name="your_tool_name",
    tool_output=str(result)[:200],
    latency_ms=latency_ms,
    success=True
)
state["trajectory"].add_step(step)
```

**Change 3 — Add loop prevention to your routing logic:**

```python
def should_continue(state: AgentState) -> str:
    # Add this to any existing routing function
    MAX_STEPS = 8  # adjust for your agent's complexity
    if state["step_count"] >= MAX_STEPS:
        state["trajectory"].loop_detected = True
        return "synthesize"  # or your final node name

    # ... rest of your existing routing logic
```

---

## ✅ Day 9 Checklist

- [ ] `trajectory_recorder.py` imports without errors
- [ ] `benchmarked_agent.py` runs and produces trajectory summaries
- [ ] `step_quality_evaluator.py` evaluates individual steps
- [ ] `run_agentic_benchmark.py` produces the aggregate report
- [ ] Loop prevention is active — agent cannot exceed MAX_STEPS
- [ ] You understand trajectory efficiency score
- [ ] Tool call accuracy is above 0.85 for your agent
- [ ] No red flags in step quality evaluation
- [ ] Trajectories saved to `trajectories.json`
- [ ] All runs visible in LangSmith with step breakdown

---

## 🎯 Interview Bits — Day 9

**Q: Why are end-to-end quality metrics insufficient for evaluating AI agents?**
*End-to-end metrics measure only the final answer. Agents produce trajectories — sequences of reasoning steps and tool calls. A correct final answer via a broken trajectory means your agent got lucky. Step-level evaluation reveals whether each intermediate result is accurate and whether errors are compounding through the reasoning chain.*

**Q: What is trajectory efficiency and how do you improve it?**
*Trajectory efficiency is the ratio of minimum necessary steps to actual steps taken. A score of 0.5 means the agent took twice as many steps as needed. Improve it by making tools more comprehensive so one call answers more, improving the system prompt to guide step selection, and adding memory so the agent does not re-retrieve information it already has.*

**Q: How do you prevent reasoning loops in LangGraph agents?**
*Set a maximum step count in the routing function and force the agent to synthesize when exceeded. Additionally track tool call history and detect when the same tool is called three or more times — this is almost always a loop. Log loop detection as a trajectory flag so you can analyze which query types trigger loops.*

**Q: What is tool call accuracy and what causes it to be low?**
*Tool call accuracy is the fraction of tool calls that succeed — correct format, valid parameters, no runtime errors. Low accuracy is caused by poorly defined tool schemas, insufficient examples in tool descriptions, or queries that fall outside tool capabilities. Improve by adding explicit parameter examples to tool docstrings and validating inputs before execution.*

**Q: How would you benchmark an agent that uses external APIs as tools?**
*Mock the external APIs during benchmarking to eliminate network variability and cost. Benchmark the agent's decision-making — which tools it chooses and how it uses results — separately from the API performance itself. Use recorded real API responses as fixtures so the agent sees realistic data without making live calls.*

---

*Tomorrow we make sure improvements stay improved.*
*You fixed the latency. You improved the quality. You optimized the cost.*
*What happens when someone merges a change that breaks all three?*
*Day 10 — regression testing and CI/CD for AI systems.*

