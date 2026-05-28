"""
Prompt registry — version-controlled store for every prompt tested.
Every benchmark run is associated with the exact prompt hash used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class PromptVersion:
    name: str
    system_prompt: str
    description: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    prompt_hash: str = ""
    faithfulness: Optional[float] = None
    context_recall: Optional[float] = None
    answer_relevancy: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.prompt_hash:
            self.prompt_hash = hashlib.md5(
                self.system_prompt.encode()
            ).hexdigest()[:8]

    def has_scores(self) -> bool:
        return self.faithfulness is not None

    def composite_score(self) -> float:
        if not self.has_scores():
            return 0.0
        return (
            (self.faithfulness or 0) * 0.4
            + (self.context_recall or 0) * 0.4
            + (self.answer_relevancy or 0) * 0.2
        )

    def summary(self) -> str:
        scores = (
            f"\n  Faithfulness:    {self.faithfulness:.3f}"
            f"\n  Context Recall:  {self.context_recall:.3f}"
            f"\n  Ans Relevancy:   {self.answer_relevancy:.3f}"
            f"\n  Avg Latency:     {self.avg_latency_ms:.0f}ms"
            if self.has_scores()
            else "\n  Not yet tested"
        )
        return (
            f"\n{'─'*50}"
            f"\n📝 {self.name}  [hash: {self.prompt_hash}]"
            f"\n  Created:   {self.created_at[:19]}"
            f"\n  Change:    {self.description}"
            f"{scores}"
        )


class PromptRegistry:

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or DATA_DIR / "prompt_registry.json"
        self.versions: list[PromptVersion] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                self.versions = [PromptVersion(**v) for v in json.load(f)]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([asdict(v) for v in self.versions], f, indent=2)

    def register(self, version: PromptVersion) -> PromptVersion:
        existing_hashes = {v.prompt_hash for v in self.versions}
        if version.prompt_hash in existing_hashes:
            print(f"⚠️  Prompt already registered: {version.prompt_hash}")
            return version
        self.versions.append(version)
        self._save()
        print(f"✅ Registered: {version.name}  [hash: {version.prompt_hash}]")
        return version

    def update_scores(
        self,
        name: str,
        faithfulness: float,
        context_recall: float,
        answer_relevancy: float,
        avg_latency_ms: float,
        notes: str = "",
    ) -> None:
        for v in self.versions:
            if v.name == name:
                v.faithfulness = faithfulness
                v.context_recall = context_recall
                v.answer_relevancy = answer_relevancy
                v.avg_latency_ms = avg_latency_ms
                v.notes = notes
                self._save()
                return
        print(f"❌ Version '{name}' not found in registry")

    def get(self, name: str) -> Optional[PromptVersion]:
        return next((v for v in self.versions if v.name == name), None)

    def print_history(self) -> None:
        if not self.versions:
            print("No prompt versions registered yet.")
            return
        print(f"\n{'━'*50}")
        print(f"PROMPT HISTORY  ({len(self.versions)} versions)")
        print(f"{'━'*50}")
        for v in self.versions:
            print(v.summary())
        tested = [v for v in self.versions if v.has_scores()]
        if tested:
            best = max(tested, key=lambda v: v.composite_score())
            print(f"\n🏆 Best version: {best.name}  "
                  f"(composite={best.composite_score():.3f})")
        print(f"{'━'*50}\n")
