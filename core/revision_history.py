"""Revision history helpers for conversational pipeline edits."""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

from core.diff_utils import diff_mapping
from core.pipeline_state import CANONICAL_STAGE_ORDER, PipelineRunRecord, PipelineState


class RevisionHistoryManager:
    """Manage revision creation, lookup, and undo targets."""

    def commit_run(
        self,
        state: PipelineState,
        *,
        revision_reason: str | None,
        changed_stages: list[str],
        changed_configs: dict | None = None,
    ) -> PipelineRunRecord:
        """Snapshot the current runtime state as a revision."""
        record = state.build_run_record(
            revision_reason=revision_reason,
            changed_stages=changed_stages,
            changed_configs=changed_configs,
            parent_run_id=state.current_run_id,
        )
        state.revision_history.append(record)
        return record

    def get_run(self, state: PipelineState, run_id: str) -> Optional[PipelineRunRecord]:
        """Return a run by id."""
        for record in state.revision_history:
            if record.run_id == run_id:
                return record
        return None

    def previous_run(self, state: PipelineState) -> Optional[PipelineRunRecord]:
        """Return the previous run relative to the current one."""
        if len(state.revision_history) < 2:
            return None
        return state.revision_history[-2]

    def restore_stage_configs(self, state: PipelineState, record: PipelineRunRecord) -> None:
        """Restore the stage configs from a prior run."""
        state.stage_configs = deepcopy(record.stage_configs)
        state.pending_revision_plan = None

    def changed_stages_between(
        self,
        older: PipelineRunRecord,
        newer: PipelineRunRecord,
    ) -> list[str]:
        """Return the canonical stages whose config changed."""
        changed: list[str] = []
        for stage in CANONICAL_STAGE_ORDER:
            if diff_mapping(older.stage_configs.get(stage, {}), newer.stage_configs.get(stage, {})):
                changed.append(stage)
        return changed
