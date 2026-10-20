"""
Independent AST and logic verifier for relational benchmark artifacts.

Audits serialized benchmark JSONL files on disk, verifies structural schema
conformance without external dependencies, and re-derives all ground-truth
epistemic classifications directly from premise graphs.
"""

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from src.symbolic.relations import (
    DirectionalRelation,
    Modality,
    Premise,
    Query,
)
from src.symbolic.solver import SymbolicSolver

VALID_TOPOLOGIES = {
    "forward_chain",
    "reverse_chain",
    "underdetermined_fork",
    "unimodal_visual",
    "unimodal_text",
}

VALID_STATUSES = {
    "entailed",
    "contradicted",
    "underdetermined",
}


@dataclass
class VerificationReport:
    """Detailed audit report of a benchmark artifact."""
    total_samples: int = 0
    valid_samples: int = 0
    status_match_count: int = 0
    extensions_match_count: int = 0
    topology_counts: dict[str, int] = field(default_factory=dict)
    status_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if all samples are valid and zero discrepancies exist."""
        return (
            self.total_samples > 0
            and self.valid_samples == self.total_samples
            and self.status_match_count == self.total_samples
            and self.extensions_match_count == self.total_samples
            and len(self.errors) == 0
        )


class BenchmarkVerifier:
    """
    Independent verification engine for relational reasoning datasets.

    Reconstructs premises from raw JSONL representations and verifies
    100% concordance against the ground-truth dual-algorithm solver.
    """

    @classmethod
    def verify_sample(cls, sample: Any, line_num: int = 1) -> list[str]:
        """
        Perform Tier 1 (Syntactic) and Tier 2 (Semantic) verification on a sample.

        Returns a list of error strings (empty if completely valid).
        """
        errors: list[str] = []

        if not isinstance(sample, dict):
            return [f"Line {line_num}: Record is not a valid JSON object."]

        # Check required top-level keys
        required_top = {"sample_id", "is_synthetic_demo", "topology", "symbolic_ground_truth"}
        missing_top = required_top - set(sample.keys())
        if missing_top:
            errors.append(f"Line {line_num}: Missing top-level keys: {sorted(missing_top)}")

        # Validate sample_id
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            errors.append(f"Line {line_num}: 'sample_id' must be a non-empty string.")

        # Validate is_synthetic_demo
        is_synthetic = sample.get("is_synthetic_demo")
        if not isinstance(is_synthetic, bool):
            errors.append(f"Line {line_num}: 'is_synthetic_demo' must be a boolean.")

        # Validate topology
        topology = sample.get("topology")
        if topology not in VALID_TOPOLOGIES:
            errors.append(f"Line {line_num}: Invalid topology: {topology!r}.")

        # Validate symbolic_ground_truth
        sgt = sample.get("symbolic_ground_truth")
        if not isinstance(sgt, dict):
            errors.append(f"Line {line_num}: 'symbolic_ground_truth' must be an object.")
            return errors

        required_sgt = {"relation", "premises", "query", "epistemic_status"}
        missing_sgt = required_sgt - set(sgt.keys())
        if missing_sgt:
            errors.append(f"Line {line_num}: Missing ground truth keys: {sorted(missing_sgt)}")

        # Relation must be left_of
        rel = sgt.get("relation")
        if rel != DirectionalRelation.LEFT_OF.value:
            errors.append(f"Line {line_num}: 'relation' must be 'left_of', got {rel!r}.")

        # Epistemic status check
        status = sgt.get("epistemic_status")
        if status not in VALID_STATUSES:
            errors.append(f"Line {line_num}: Invalid epistemic_status: {status!r}.")

        # Premises array check (minItems: 2)
        premises_raw = sgt.get("premises")
        if not isinstance(premises_raw, list):
            errors.append(f"Line {line_num}: 'premises' must be a list.")
            return errors

        if len(premises_raw) < 2:
            errors.append(
                f"Line {line_num}: 'premises' must have at least 2 items, found {len(premises_raw)}."
            )

        parsed_premises: list[Premise] = []
        for p_idx, p in enumerate(premises_raw):
            if not isinstance(p, dict):
                errors.append(f"Line {line_num}: Premise {p_idx} is not an object.")
                continue
            req_p = {"modality", "subject", "object", "relation"}
            missing_p = req_p - set(p.keys())
            if missing_p:
                errors.append(f"Line {line_num}: Premise {p_idx} missing keys: {sorted(missing_p)}")
                continue

            subj = p.get("subject", "")
            obj = p.get("object", "")
            mod = p.get("modality", "")
            p_rel = p.get("relation", "")

            if not isinstance(subj, str) or not subj.strip():
                errors.append(f"Line {line_num}: Premise {p_idx} subject must be non-empty string.")
            if not isinstance(obj, str) or not obj.strip():
                errors.append(f"Line {line_num}: Premise {p_idx} object must be non-empty string.")
            if subj == obj and subj:
                errors.append(f"Line {line_num}: Premise {p_idx} irreflexivity violation ({subj} == {obj}).")
            if p_rel != DirectionalRelation.LEFT_OF.value:
                errors.append(f"Line {line_num}: Premise {p_idx} relation must be 'left_of', got {p_rel!r}.")
            if mod not in {"visual", "textual"}:
                errors.append(f"Line {line_num}: Premise {p_idx} modality must be 'visual' or 'textual'.")

            try:
                parsed_premises.append(
                    Premise(
                        subject=subj,
                        object=obj,
                        relation=DirectionalRelation.LEFT_OF,
                        modality=Modality(mod) if mod in {"visual", "textual"} else Modality.TEXTUAL,
                    )
                )
            except Exception as e:
                errors.append(f"Line {line_num}: Failed to construct Premise {p_idx}: {e}")

        # Query object check
        query_raw = sgt.get("query")
        if not isinstance(query_raw, dict):
            errors.append(f"Line {line_num}: 'query' must be an object.")
            return errors

        q_subj = query_raw.get("subject", "")
        q_obj = query_raw.get("object", "")
        q_rel = query_raw.get("relation", "")

        if not isinstance(q_subj, str) or not q_subj.strip():
            errors.append(f"Line {line_num}: Query subject must be non-empty string.")
        if not isinstance(q_obj, str) or not q_obj.strip():
            errors.append(f"Line {line_num}: Query object must be non-empty string.")
        if q_subj == q_obj and q_subj:
            errors.append(f"Line {line_num}: Query irreflexivity violation ({q_subj} == {q_obj}).")
        if q_rel != DirectionalRelation.LEFT_OF.value:
            errors.append(f"Line {line_num}: Query relation must be 'left_of', got {q_rel!r}.")

        # If syntactic errors exist, abort semantic re-derivation
        if errors:
            return errors

        # Tier 2 Semantic Verification: Re-derive ground truth with solver
        try:
            parsed_query = Query(subject=q_subj, object=q_obj, relation=DirectionalRelation.LEFT_OF)
            solver_res = SymbolicSolver.solve(parsed_premises, parsed_query, verify_with_oracle=True)

            # Check epistemic status match
            if solver_res.epistemic_status.value != status:
                errors.append(
                    f"Line {line_num}: Epistemic status discrepancy for sample {sample_id!r}! "
                    f"Recorded: {status!r}, Re-derived: {solver_res.epistemic_status.value!r}."
                )

            # Check linear extensions match if present
            recorded_exts = sgt.get("consistent_linear_extensions")
            if recorded_exts is not None:
                # Compare sorted linear extensions
                recorded_sorted = sorted([tuple(x) for x in recorded_exts])
                derived_sorted = sorted([tuple(x) for x in solver_res.linear_extensions])
                if recorded_sorted != derived_sorted:
                    errors.append(
                        f"Line {line_num}: Linear extensions mismatch for sample {sample_id!r}! "
                        f"Recorded {len(recorded_sorted)} extensions, Re-derived {len(derived_sorted)}."
                    )

        except Exception as exc:
            errors.append(f"Line {line_num}: Semantic deduction raised unexpected error: {exc}")

        return errors

    @classmethod
    def verify_file(cls, filepath: str | Path) -> VerificationReport:
        """
        Verify all samples in a JSONL benchmark file on disk.

        Args:
            filepath: Path to the JSONL file.

        Returns:
            VerificationReport summarizing findings.
        """
        path = Path(filepath)
        report = VerificationReport()

        if not path.is_file():
            report.errors.append(f"File not found: {path}")
            return report

        topology_counter: Counter[str] = Counter()
        status_counter: Counter[str] = Counter()
        split_counter: Counter[str] = Counter()

        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                raw_line = line.strip()
                if not raw_line:
                    continue

                report.total_samples += 1

                try:
                    sample = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    report.errors.append(f"Line {line_idx}: Invalid JSON: {exc}")
                    continue

                sample_errors = cls.verify_sample(sample, line_num=line_idx)
                if sample_errors:
                    report.errors.extend(sample_errors)
                else:
                    report.valid_samples += 1
                    report.status_match_count += 1
                    report.extensions_match_count += 1

                    top = sample.get("topology", "unknown")
                    stat = sample.get("symbolic_ground_truth", {}).get("epistemic_status", "unknown")
                    spl = sample.get("split", "unknown")

                    topology_counter[top] += 1
                    status_counter[stat] += 1
                    split_counter[spl] += 1

        report.topology_counts = dict(topology_counter)
        report.status_counts = dict(status_counter)
        report.split_counts = dict(split_counter)

        return report
