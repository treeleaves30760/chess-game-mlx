"""Ponder manager — tracks multi-ponder UI state.

Receives PonderProgress / PonderHit / PonderMiss events from the engine
client and provides a clean snapshot for the analysis panel to render.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PonderTreeState:
    """State for a single ponder tree (one opponent move hypothesis)."""

    tree_id: int = 0
    opponent_move: str = ""
    depth: int = 0
    score_cp: int = 0
    pv: list[str] = field(default_factory=list)
    # Weight / budget fraction
    budget_weight: float = 0.0
    # Whether this tree is the current "active" ponder hypothesis
    is_active: bool = True
    # Timestamp of last update
    last_updated: float = field(default_factory=time.monotonic)


@dataclass
class PolicyMovePreview:
    """A single opponent move from a policy_preview event."""

    uci: str = ""
    prob: float = 0.0


class PonderManager:
    """Manages multi-ponder state across engine events.

    The main loop calls update_from_event() with each EngineEvent;
    the analysis panel reads trees_snapshot() for rendering.
    """

    DEFAULT_WEIGHTS = [0.40, 0.20, 0.15, 0.15, 0.10]

    def __init__(self) -> None:
        self._trees: dict[int, PonderTreeState] = {}
        self._policy_preview: list[PolicyMovePreview] = []
        self._hit_tree: int | None = None
        self._hit_bestmove: str | None = None
        self._hit_score_cp: int = 0
        self._is_pondering: bool = False
        self._trees_discarded: int = 0

    # -----------------------------------------------------------------------
    # Event ingestion
    # -----------------------------------------------------------------------

    def on_policy_preview(self, moves: list[dict[str, Any]]) -> None:
        """Handle policy_preview notification."""
        self._policy_preview = [
            PolicyMovePreview(uci=m.get("uci", ""), prob=m.get("prob", 0.0))
            for m in moves
        ]
        self._is_pondering = True
        self._hit_tree = None

    def on_ponder_progress(self, trees_data: list[dict[str, Any]]) -> None:
        """Handle ponder_progress notification."""
        weights = self.DEFAULT_WEIGHTS
        for td in trees_data:
            tid = td.get("tree", 0)
            weight = weights[tid] if tid < len(weights) else 0.0
            state = PonderTreeState(
                tree_id=tid,
                opponent_move=td.get("opponent_move", ""),
                depth=td.get("depth", 0),
                score_cp=td.get("score_cp", 0),
                pv=td.get("pv", []),
                budget_weight=weight,
                is_active=True,
                last_updated=time.monotonic(),
            )
            self._trees[tid] = state

    def on_ponder_hit(self, tree_id: int, bestmove: str, score_cp: int) -> None:
        """Handle ponder_hit notification."""
        self._hit_tree = tree_id
        self._hit_bestmove = bestmove
        self._hit_score_cp = score_cp
        self._is_pondering = False
        # Mark all other trees as inactive
        for tid, tree in self._trees.items():
            tree.is_active = tid == tree_id

    def on_ponder_miss(self, trees_discarded: int) -> None:
        """Handle ponder_miss notification."""
        self._trees.clear()
        self._policy_preview = []
        self._hit_tree = None
        self._is_pondering = False
        self._trees_discarded = trees_discarded

    def reset(self) -> None:
        """Clear all ponder state (e.g. on new game)."""
        self._trees.clear()
        self._policy_preview = []
        self._hit_tree = None
        self._hit_bestmove = None
        self._is_pondering = False
        self._trees_discarded = 0

    # -----------------------------------------------------------------------
    # Read accessors (called by the render loop)
    # -----------------------------------------------------------------------

    @property
    def is_pondering(self) -> bool:
        return self._is_pondering

    @property
    def hit_tree(self) -> int | None:
        return self._hit_tree

    @property
    def hit_bestmove(self) -> str | None:
        return self._hit_bestmove

    @property
    def hit_score_cp(self) -> int:
        return self._hit_score_cp

    def trees_snapshot(self) -> list[PonderTreeState]:
        """Return a sorted list of ponder trees for rendering."""
        return sorted(self._trees.values(), key=lambda t: -t.budget_weight)

    def policy_preview_snapshot(self) -> list[PolicyMovePreview]:
        return list(self._policy_preview)
