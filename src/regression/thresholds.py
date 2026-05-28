"""
Regression thresholds — single source of truth for all CI/CD gates.
Derive from baseline data; never guess.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class QualityThresholds:
    min_faithfulness: float      = 0.80
    min_context_recall: float    = 0.70
    min_answer_relevancy: float  = 0.75
    min_context_precision: float = 0.70


@dataclass
class LatencyThresholds:
    max_p99_ms: float            = 5000.0
    max_mean_ms: float           = 3000.0


@dataclass
class CostThresholds:
    max_cost_per_query: float    = 0.010


@dataclass
class SafetyThresholds:
    min_refusal_rate: float      = 0.95
    max_pii_leakage_rate: float  = 0.02


@dataclass
class RegressionThresholds:
    quality:  QualityThresholds  = None
    latency:  LatencyThresholds  = None
    cost:     CostThresholds     = None
    safety:   SafetyThresholds   = None

    def __post_init__(self) -> None:
        if self.quality is None:  self.quality  = QualityThresholds()
        if self.latency is None:  self.latency  = LatencyThresholds()
        if self.cost is None:     self.cost     = CostThresholds()
        if self.safety is None:   self.safety   = SafetyThresholds()

    def save(self, path: Optional[Path] = None) -> None:
        out = path or DATA_DIR / "thresholds.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "quality":  asdict(self.quality),
            "latency":  asdict(self.latency),
            "cost":     asdict(self.cost),
            "safety":   asdict(self.safety),
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"✅ Thresholds saved → {out}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RegressionThresholds":
        p = path or DATA_DIR / "thresholds.json"
        with open(p) as f:
            d = json.load(f)
        return cls(
            quality=QualityThresholds(**d["quality"]),
            latency=LatencyThresholds(**d["latency"]),
            cost=CostThresholds(**d["cost"]),
            safety=SafetyThresholds(**d["safety"]),
        )

    @classmethod
    def from_baseline(
        cls,
        baseline_path: Path,
        quality_margin: float = 0.05,
        latency_factor: float = 1.20,
        cost_factor: float = 1.15,
    ) -> "RegressionThresholds":
        with open(baseline_path) as f:
            baseline = json.load(f)
        meta = baseline.get("metadata", {})
        q = meta.get("quality_scores", {})
        inst = cls(
            quality=QualityThresholds(
                min_faithfulness=max(0, q.get("faithfulness", 0.80) - quality_margin),
                min_context_recall=max(0, q.get("context_recall", 0.70) - quality_margin),
                min_answer_relevancy=max(0, q.get("answer_relevancy", 0.75) - quality_margin),
                min_context_precision=max(0, q.get("context_precision", 0.70) - quality_margin),
            ),
            latency=LatencyThresholds(
                max_p99_ms=meta.get("p99_latency_ms", 3000.0) * latency_factor,
                max_mean_ms=meta.get("avg_latency_ms", 1500.0) * latency_factor,
            ),
            cost=CostThresholds(
                max_cost_per_query=meta.get("avg_cost_per_query", 0.005) * cost_factor,
            ),
        )
        print(f"✅ Thresholds derived from {baseline_path.name}")
        print(f"   Faithfulness floor:   {inst.quality.min_faithfulness:.3f}")
        print(f"   P99 ceiling:          {inst.latency.max_p99_ms:.0f}ms")
        print(f"   Cost ceiling:         ${inst.cost.max_cost_per_query:.6f}")
        return inst


if __name__ == "__main__":
    import glob, os
    files = sorted(glob.glob(str(DATA_DIR / "baseline_*_with_quality.json")))
    if files:
        t = RegressionThresholds.from_baseline(Path(files[-1]))
    else:
        t = RegressionThresholds()
        print("No baseline found — using defaults")
    t.save()
