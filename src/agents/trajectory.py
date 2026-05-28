"""
Trajectory recorder — captures every step an agent takes.
Computes trajectory length, tool accuracy, efficiency, and loop detection.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class AgentStep:
    step_number: int
    step_type: str          # reasoning | tool_call | synthesis
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    reasoning: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        status = "✓" if self.success else "✗"
        tool = f" → {self.tool_name}" if self.tool_name else ""
        err = f" [ERR: {self.error}]" if self.error else ""
        return f"  Step {self.step_number}: {status} {self.step_type}{tool} ({self.latency_ms:.0f}ms){err}"


@dataclass
class AgentTrajectory:
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    success: bool = True
    trajectory_length: int = 0
    tool_call_accuracy: float = 0.0
    trajectory_efficiency: float = 0.0
    loop_detected: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)
        self.total_latency_ms += step.latency_ms
        self.total_tokens += step.tokens_used

    def compute_metrics(self, min_steps_expected: int = 2) -> None:
        self.trajectory_length = len(self.steps)
        tool_steps = [s for s in self.steps if s.step_type == "tool_call"]
        if tool_steps:
            ok = sum(1 for s in tool_steps if s.success)
            self.tool_call_accuracy = ok / len(tool_steps)
        else:
            self.tool_call_accuracy = 1.0
        if self.trajectory_length > 0:
            self.trajectory_efficiency = min(
                min_steps_expected / self.trajectory_length, 1.0
            )
        tool_names = [s.tool_name for s in tool_steps if s.tool_name]
        if tool_names:
            counts = Counter(tool_names)
            self.loop_detected = any(c >= 3 for c in counts.values())

    def summary(self) -> str:
        loop = " ⚠️  LOOP" if self.loop_detected else ""
        lines = [
            f"\n{'━'*55}",
            f"🤖 AGENT TRAJECTORY",
            f"{'━'*55}",
            f"Query:      {self.query[:50]}{'...' if len(self.query) > 50 else ''}",
            f"Steps:      {self.trajectory_length}{loop}",
            f"Efficiency: {self.trajectory_efficiency:.2f}  (1.0 = optimal)",
            f"Tool Acc:   {self.tool_call_accuracy:.2f}",
            f"Latency:    {self.total_latency_ms:.0f}ms",
            f"{'─'*55}",
            "Steps:",
        ]
        lines += [s.summary() for s in self.steps]
        lines += [
            f"{'─'*55}",
            f"Answer: {self.final_answer[:80]}{'...' if len(self.final_answer) > 80 else ''}",
            f"{'━'*55}",
        ]
        return "\n".join(lines)


class TrajectoryRecorder:

    def __init__(self) -> None:
        self.trajectories: list[AgentTrajectory] = []

    def new_trajectory(self, query: str) -> AgentTrajectory:
        return AgentTrajectory(query=query)

    def record(self, trajectory: AgentTrajectory) -> None:
        self.trajectories.append(trajectory)

    def print_report(self) -> None:
        if not self.trajectories:
            print("No trajectories recorded.")
            return
        lengths = [t.trajectory_length for t in self.trajectories]
        efficiencies = [t.trajectory_efficiency for t in self.trajectories]
        accs = [t.tool_call_accuracy for t in self.trajectories]
        latencies = [t.total_latency_ms for t in self.trajectories]
        loops = sum(1 for t in self.trajectories if t.loop_detected)
        failures = sum(1 for t in self.trajectories if not t.success)

        print(f"\n{'━'*55}")
        print(f"AGENTIC BENCHMARK REPORT  ({len(self.trajectories)} trajectories)")
        print(f"{'━'*55}")
        print(f"  Avg steps:         {np.mean(lengths):.1f}")
        print(f"  Max steps:         {max(lengths)}")
        print(f"  Avg efficiency:    {np.mean(efficiencies):.2f}")
        print(f"  Avg tool accuracy: {np.mean(accs):.2f}")
        print(f"  Avg latency:       {np.mean(latencies):.0f}ms")
        print(f"  Loops detected:    {loops}/{len(self.trajectories)}")
        print(f"  Failures:          {failures}/{len(self.trajectories)}")
        print(f"{'─'*55}")

        avg_eff = np.mean(efficiencies)
        avg_acc = np.mean(accs)
        if avg_eff >= 0.8 and avg_acc >= 0.9 and loops == 0:
            print("  Status: ✅ Agent performing well")
        elif avg_eff >= 0.6 and avg_acc >= 0.75:
            print("  Status: 🟡 Agent needs optimization")
        else:
            print("  Status: 🔴 Agent has significant issues")
        print(f"{'━'*55}\n")

    def save(self, path: Optional[Path] = None) -> None:
        out = path or DATA_DIR / "trajectories.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "query": t.query,
                "trajectory_length": t.trajectory_length,
                "tool_call_accuracy": t.tool_call_accuracy,
                "trajectory_efficiency": t.trajectory_efficiency,
                "loop_detected": t.loop_detected,
                "total_latency_ms": t.total_latency_ms,
                "success": t.success,
                "final_answer": t.final_answer[:200],
                "steps": [
                    {
                        "step_number": s.step_number,
                        "step_type": s.step_type,
                        "tool_name": s.tool_name,
                        "latency_ms": s.latency_ms,
                        "success": s.success,
                        "error": s.error,
                    }
                    for s in t.steps
                ],
            }
            for t in self.trajectories
        ]
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Trajectories saved → {out}")
