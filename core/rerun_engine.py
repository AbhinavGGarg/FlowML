"""Dependency-aware rerun support for pipeline revisions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from core.pipeline_state import (
    PipelineState,
    canonical_downstream_stages,
    concrete_stages_from_canonical,
    normalize_stage_name,
)


logger = logging.getLogger(__name__)


class DependencyAwareRerunEngine:
    """Resolve and run downstream stages after a controlled revision."""

    def resolve_canonical_stages(self, rerun_from_stage: str) -> list[str]:
        """Return canonical stages that should be considered rerun targets."""
        return canonical_downstream_stages(rerun_from_stage)

    def resolve_concrete_stages(self, rerun_from_stage: str) -> list[str]:
        """Return concrete runtime stages that must be rerun."""
        return concrete_stages_from_canonical(rerun_from_stage)

    def reset_downstream_state(
        self,
        state: PipelineState,
        rerun_from_stage: str,
        canonical_rerun_stages: list[str],
        rerun_stages: list[str],
    ) -> None:
        """Clear downstream statuses and outputs before rerunning."""
        canonical_start = normalize_stage_name(rerun_from_stage)
        restart_message = (
            "Revision rerun requested from canonical stage "
            f"`{canonical_start}`. Canonical stages: {', '.join(canonical_rerun_stages)}. "
            f"Concrete runtime stages: {', '.join(rerun_stages)}."
        )
        for stage in rerun_stages:
            state.stage_statuses[stage] = "waiting"
            state.stage_results.pop(stage, None)
            state.stage_logs.setdefault(stage, [])
            state.stage_logs[stage].append(restart_message)
            state.stage_logs[stage].append("Revision agent marked this stage for rerun.")

        if "results" in rerun_stages:
            state.stage_results.pop("explanation", None)

    async def rerun(
        self,
        *,
        state: PipelineState,
        rerun_from_stage: str,
        config: Any,
        stage_runner: Callable[[str, Any], Awaitable[None]],
    ) -> list[str]:
        """Rerun affected stages from the chosen starting point."""
        canonical_start = normalize_stage_name(rerun_from_stage)
        canonical_rerun_stages = self.resolve_canonical_stages(canonical_start)
        rerun_stages = self.resolve_concrete_stages(canonical_start)
        logger.info(
            "Revision rerun starting from canonical_stage=%s | canonical_rerun=%s | concrete_rerun=%s",
            canonical_start,
            canonical_rerun_stages,
            rerun_stages,
        )
        self.reset_downstream_state(state, canonical_start, canonical_rerun_stages, rerun_stages)
        for stage in rerun_stages:
            await asyncio.sleep(0)
            await stage_runner(stage, config)
            await asyncio.sleep(0.2)
        logger.info(
            "Revision rerun completed from canonical_stage=%s | concrete_rerun=%s",
            canonical_start,
            rerun_stages,
        )
        return rerun_stages
