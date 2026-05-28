"""
Drift detector — detects when production query distribution shifts away from eval baseline.
Uses Evidently AI for statistical feature drift detection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.report import Report

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def extract_features(queries: list[str]) -> pd.DataFrame:
    rows = []
    for q in queries:
        words = q.lower().split()
        rows.append({
            "query_length":     len(q),
            "word_count":       len(words),
            "avg_word_length":  np.mean([len(w) for w in words]) if words else 0,
            "question_words":   sum(1 for w in words if w in ["what","how","why","when","where","who","which"]),
            "analysis_words":   sum(1 for w in words if w in ["analyze","evaluate","compare","design","recommend"]),
            "technical_terms":  sum(1 for w in words if w in ["rag","llm","embedding","vector","retrieval","ragas","faiss","latency","token"]),
            "is_definition":    int(any(w in words for w in ["what","define","definition"])),
            "is_howto":         int(any(w in words for w in ["how","steps","process"])),
            "is_comparison":    int(any(w in words for w in ["compare","difference","versus","vs"])),
        })
    return pd.DataFrame(rows)


@dataclass
class QualityWindow:
    label: str
    faithfulness_scores: list[float]
    relevancy_scores: list[float]
    latencies: list[float]

    def to_dict(self) -> dict:
        return {
            "window": self.label,
            "avg_faithfulness": np.mean(self.faithfulness_scores),
            "avg_relevancy":    np.mean(self.relevancy_scores),
            "p99_latency":      np.percentile(self.latencies, 99),
            "sample_count":     len(self.faithfulness_scores),
        }


class DriftDetector:

    def __init__(self, reference_queries: list[str]) -> None:
        self.reference_df = extract_features(reference_queries)
        self.quality_windows: list[QualityWindow] = []
        print(f"  Drift detector ready ({len(reference_queries)} reference queries)")

    def detect_query_drift(
        self, production_queries: list[str], label: str = "current"
    ) -> dict:
        print(f"\n  Query drift detection on {len(production_queries)} production queries...")
        production_df = extract_features(production_queries)

        report = Report(metrics=[
            DatasetDriftMetric(),
            ColumnDriftMetric(column_name="query_length"),
            ColumnDriftMetric(column_name="word_count"),
            ColumnDriftMetric(column_name="analysis_words"),
            ColumnDriftMetric(column_name="technical_terms"),
        ])
        report.run(reference_data=self.reference_df, current_data=production_df)
        rd = report.as_dict()

        ds = rd["metrics"][0]["result"]
        drift_detected = ds.get("dataset_drift", False)
        drift_share    = ds.get("share_of_drifted_columns", 0)

        col_drifts = {}
        for m in rd["metrics"][1:]:
            col = m["result"].get("column_name", "?")
            col_drifts[col] = {
                "drifted": m["result"].get("drift_detected", False),
                "score":   m["result"].get("drift_score", 0),
            }

        result = {
            "window": label, "drift_detected": drift_detected,
            "drift_share": drift_share, "column_drifts": col_drifts,
            "timestamp": datetime.now().isoformat(),
        }

        html_path = DATA_DIR / f"drift_report_{label}.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(html_path))

        self._print_drift(result)
        print(f"  Full HTML report → {html_path}")
        return result

    def track_quality_window(self, window: QualityWindow) -> None:
        self.quality_windows.append(window)

    def detect_quality_drift(self, threshold: float = 0.05) -> dict:
        if len(self.quality_windows) < 2:
            return {"message": "Need ≥ 2 windows"}
        base, latest = self.quality_windows[0], self.quality_windows[-1]
        f_drop = np.mean(base.faithfulness_scores) - np.mean(latest.faithfulness_scores)
        r_drop = np.mean(base.relevancy_scores)    - np.mean(latest.relevancy_scores)
        result = {
            "faithfulness_drift": f_drop > threshold,
            "faithfulness_drop":  round(f_drop, 3),
            "relevancy_drift":    r_drop > threshold,
            "relevancy_drop":     round(r_drop, 3),
            "alert":              f_drop > threshold or r_drop > threshold,
        }
        print(f"\n{'━'*50}")
        print(f"QUALITY DRIFT  ({base.label} → {latest.label})")
        print(f"{'━'*50}")
        print(f"  Faithfulness: {np.mean(base.faithfulness_scores):.3f} → "
              f"{np.mean(latest.faithfulness_scores):.3f}  "
              f"{'⚠️  DRIFT' if result['faithfulness_drift'] else '✅ stable'}")
        print(f"  Relevancy:    {np.mean(base.relevancy_scores):.3f} → "
              f"{np.mean(latest.relevancy_scores):.3f}  "
              f"{'⚠️  DRIFT' if result['relevancy_drift'] else '✅ stable'}")
        if result["alert"]:
            print("  🔴 Quality drift detected — investigate recent changes")
        else:
            print("  🟢 Quality stable across windows")
        print(f"{'━'*50}\n")
        return result

    def save_history(self, path: Optional[Path] = None) -> None:
        out = path or DATA_DIR / "drift_history.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump([w.to_dict() for w in self.quality_windows], f, indent=2)
        print(f"✅ Drift history → {out}")

    def _print_drift(self, result: dict) -> None:
        print(f"\n{'━'*50}")
        print(f"QUERY DRIFT REPORT")
        print(f"{'━'*50}")
        print(f"  Drift detected:   {'⚠️  YES' if result['drift_detected'] else '✅ NO'}")
        print(f"  Drifted features: {result['drift_share']*100:.0f}%")
        for col, data in result.get("column_drifts", {}).items():
            flag = "⚠️ " if data["drifted"] else "✅"
            print(f"    {flag} {col:<25} score={data['score']:.3f}")
        if result["drift_detected"]:
            print("  ⚠️  Update eval dataset to include new query patterns")
        print(f"{'━'*50}")
