"""
Token-level cost tracker — exact per-query cost, projection, and breakdown by model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tiktoken

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":               {"input_per_1k": 0.005,    "output_per_1k": 0.015},
    "gpt-4o-mini":          {"input_per_1k": 0.000150, "output_per_1k": 0.000600},
    "claude-sonnet-4-6":    {"input_per_1k": 0.003,    "output_per_1k": 0.015},
    "claude-haiku-4-5":     {"input_per_1k": 0.00025,  "output_per_1k": 0.00125},
}


@dataclass
class QueryCost:
    query: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    query_type: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CostTracker:

    def __init__(self) -> None:
        self.records: list[QueryCost] = []
        self._enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def track(
        self,
        model: str,
        input_text: str,
        output_text: str,
        query_type: str = "unknown",
    ) -> QueryCost:
        p = MODEL_PRICING.get(model, {"input_per_1k": 0.001, "output_per_1k": 0.002})
        in_tok = self.count_tokens(input_text)
        out_tok = self.count_tokens(output_text)
        in_cost = (in_tok / 1000) * p["input_per_1k"]
        out_cost = (out_tok / 1000) * p["output_per_1k"]
        record = QueryCost(
            query=input_text[:100],
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            input_cost_usd=round(in_cost, 8),
            output_cost_usd=round(out_cost, 8),
            total_cost_usd=round(in_cost + out_cost, 8),
            query_type=query_type,
        )
        self.records.append(record)
        return record

    def projection(self, queries_per_day: int) -> dict:
        if not self.records:
            return {}
        avg = float(np.mean([r.total_cost_usd for r in self.records]))
        monthly = queries_per_day * 30
        return {
            "avg_cost_per_query_usd": round(avg, 6),
            "queries_per_day": queries_per_day,
            "projected_monthly_usd": round(avg * monthly, 2),
            "projected_annual_usd": round(avg * monthly * 12, 2),
        }

    def print_report(self, queries_per_day: int = 1000) -> None:
        if not self.records:
            print("No cost records yet.")
            return
        df = pd.DataFrame([r.__dict__ for r in self.records])
        proj = self.projection(queries_per_day)
        print(f"\n{'━'*55}")
        print("COST ANALYSIS REPORT")
        print(f"{'━'*55}")
        print(f"  Queries analyzed:    {len(self.records)}")
        print(f"  Avg cost per query:  ${df['total_cost_usd'].mean():.6f}")
        print(f"  Total cost (sample): ${df['total_cost_usd'].sum():.6f}")
        print(f"  Avg input tokens:    {df['input_tokens'].mean():.0f}")
        print(f"  Avg output tokens:   {df['output_tokens'].mean():.0f}")
        print(f"{'─'*55}")
        print(f"  PROJECTION ({queries_per_day:,} queries/day)")
        print(f"  Monthly:  ${proj.get('projected_monthly_usd', 0):,.2f}")
        print(f"  Annual:   ${proj.get('projected_annual_usd', 0):,.2f}")
        if df["model"].nunique() > 1:
            print(f"{'─'*55}")
            print("  Cost by model:")
            for model, group in df.groupby("model"):
                print(f"    {model:<20} avg=${group['total_cost_usd'].mean():.6f}"
                      f"  count={len(group)}")
        print(f"{'━'*55}\n")

    def save(self, path: Optional[Path] = None) -> None:
        out = path or DATA_DIR / "cost_records.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump([r.__dict__ for r in self.records], f, indent=2)
        print(f"✅ Cost records saved → {out}")
