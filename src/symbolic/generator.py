"""
Deterministic benchmark dataset generator for relational reasoning.

Assembles balanced, logically verified benchmark suites across forward chains,
reverse chains, underdetermined forks, and unimodal controls.
"""

import json
from pathlib import Path
import random
from typing import Any, Optional, Sequence

from src.symbolic.relations import (
    DirectionalRelation,
    Modality,
    Premise,
    Query,
)
from src.symbolic.solver import SymbolicSolver


IN_DISTRIBUTION_ENTITIES = (
    "blue_cube",
    "red_sphere",
    "green_cylinder",
    "yellow_cone",
    "purple_pyramid",
    "orange_torus",
    "cyan_box",
    "magenta_prism",
)

HELD_OUT_ENTITIES = (
    "silver_hexagon",
    "amber_ring",
    "teal_wedge",
    "violet_star",
    "crimson_block",
    "emerald_oval",
    "golden_disc",
    "indigo_capsule",
)

# Formal check: Disjoint entity pools
assert set(IN_DISTRIBUTION_ENTITIES).isdisjoint(set(HELD_OUT_ENTITIES)), (
    "In-distribution and held-out entity pools must be strictly disjoint!"
)

TOPOLOGIES = (
    "forward_chain",
    "reverse_chain",
    "underdetermined_fork",
    "unimodal_visual",
    "unimodal_text",
)


class BenchmarkGenerator:
    """
    Generates reproducible relational reasoning benchmark datasets.

    Enforces strict seed control, entity pool isolation, balanced topology
    distribution, and ground-truth solver validation.
    """

    def __init__(
        self,
        seed: int = 42,
        in_distribution_entities: Sequence[str] = IN_DISTRIBUTION_ENTITIES,
        held_out_entities: Sequence[str] = HELD_OUT_ENTITIES,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.id_entities = list(in_distribution_entities)
        self.ho_entities = list(held_out_entities)

        if not set(self.id_entities).isdisjoint(set(self.ho_entities)):
            raise ValueError("Entity pools for in-distribution and held-out must be disjoint")

    def _sample_entities(self, pool: list[str], k: int = 3) -> list[str]:
        """Sample k distinct entities from the specified pool."""
        if len(pool) < k:
            raise ValueError(f"Entity pool must have at least {k} entities, found {len(pool)}")
        return self.rng.sample(pool, k)

    def generate_sample(
        self,
        sample_idx: int,
        topology: str,
        split: str = "in_distribution",
    ) -> dict[str, Any]:
        """
        Generate a single verified relational reasoning sample.

        Args:
            sample_idx: Index of the sample in the suite.
            topology: One of TOPOLOGIES.
            split: 'in_distribution' or 'held_out'.

        Returns:
            Dictionary matching the DatasetSample schema.
        """
        pool = self.id_entities if split == "in_distribution" else self.ho_entities
        ents = self._sample_entities(pool, k=3)
        a, b, c = ents[0], ents[1], ents[2]

        premises: list[Premise]
        query: Query

        # Modality assignment for cross-modal pairs (prevent positional modality bias)
        mod1, mod2 = (
            (Modality.VISUAL, Modality.TEXTUAL)
            if self.rng.choice([True, False])
            else (Modality.TEXTUAL, Modality.VISUAL)
        )

        if topology == "forward_chain":
            # R(A, B) and R(B, C) => R(A, C) [entailed]
            premises = [
                Premise(subject=a, object=b, relation=DirectionalRelation.LEFT_OF, modality=mod1),
                Premise(subject=b, object=c, relation=DirectionalRelation.LEFT_OF, modality=mod2),
            ]
            query = Query(subject=a, object=c, relation=DirectionalRelation.LEFT_OF)

        elif topology == "reverse_chain":
            # R(B, A) and R(C, B) => R(A, C) [contradicted, since C < B < A]
            premises = [
                Premise(subject=b, object=a, relation=DirectionalRelation.LEFT_OF, modality=mod1),
                Premise(subject=c, object=b, relation=DirectionalRelation.LEFT_OF, modality=mod2),
            ]
            query = Query(subject=a, object=c, relation=DirectionalRelation.LEFT_OF)

        elif topology == "underdetermined_fork":
            # Fork variants:
            # Variant 1 (fan-in): R(A, B) and R(C, B) => R(A, C) [underdetermined]
            # Variant 2 (fan-out): R(B, A) and R(B, C) => R(A, C) [underdetermined]
            is_fan_in = self.rng.choice([True, False])
            if is_fan_in:
                premises = [
                    Premise(subject=a, object=b, relation=DirectionalRelation.LEFT_OF, modality=mod1),
                    Premise(subject=c, object=b, relation=DirectionalRelation.LEFT_OF, modality=mod2),
                ]
            else:
                premises = [
                    Premise(subject=b, object=a, relation=DirectionalRelation.LEFT_OF, modality=mod1),
                    Premise(subject=b, object=c, relation=DirectionalRelation.LEFT_OF, modality=mod2),
                ]
            query = Query(subject=a, object=c, relation=DirectionalRelation.LEFT_OF)

        elif topology == "unimodal_visual":
            # Control: all visual premises
            is_positive = self.rng.choice([True, False])
            premises = [
                Premise(subject=a, object=b, relation=DirectionalRelation.LEFT_OF, modality=Modality.VISUAL),
                Premise(subject=b, object=c, relation=DirectionalRelation.LEFT_OF, modality=Modality.VISUAL),
            ]
            if is_positive:
                query = Query(subject=a, object=c, relation=DirectionalRelation.LEFT_OF)
            else:
                query = Query(subject=c, object=a, relation=DirectionalRelation.LEFT_OF)

        elif topology == "unimodal_text":
            # Control: all textual premises
            is_positive = self.rng.choice([True, False])
            premises = [
                Premise(subject=a, object=b, relation=DirectionalRelation.LEFT_OF, modality=Modality.TEXTUAL),
                Premise(subject=b, object=c, relation=DirectionalRelation.LEFT_OF, modality=Modality.TEXTUAL),
            ]
            if is_positive:
                query = Query(subject=a, object=c, relation=DirectionalRelation.LEFT_OF)
            else:
                query = Query(subject=c, object=a, relation=DirectionalRelation.LEFT_OF)

        else:
            raise ValueError(f"Unknown topology: {topology!r}. Expected one of {TOPOLOGIES}")

        # Shuffle premise presentation order to prevent positional heuristic memorization
        self.rng.shuffle(premises)

        # Ground-truth validation via dual-algorithm solver
        solver_res = SymbolicSolver.solve(premises, query, verify_with_oracle=True)

        sample_id = f"sample_{split}_{sample_idx:06d}"

        return {
            "sample_id": sample_id,
            "is_synthetic_demo": False,  # Validated benchmark data, NOT mock demo
            "topology": topology,
            "split": split,
            "symbolic_ground_truth": {
                "relation": DirectionalRelation.LEFT_OF.value,
                "premises": [
                    {
                        "modality": p.modality.value,
                        "subject": p.subject,
                        "object": p.object,
                        "relation": p.relation.value,
                    }
                    for p in premises
                ],
                "query": {
                    "subject": query.subject,
                    "object": query.object,
                    "relation": query.relation.value,
                },
                "epistemic_status": solver_res.epistemic_status.value,
                "consistent_linear_extensions": solver_res.linear_extensions,
            },
        }

    def generate_dataset(
        self,
        num_samples: int = 1000,
        split: str = "in_distribution",
    ) -> list[dict[str, Any]]:
        """
        Generate a balanced relational reasoning dataset.

        Args:
            num_samples: Total number of samples to generate.
            split: 'in_distribution' or 'held_out'.

        Returns:
            List of validated sample records.
        """
        samples: list[dict[str, Any]] = []
        n_topologies = len(TOPOLOGIES)

        for i in range(num_samples):
            topology = TOPOLOGIES[i % n_topologies]
            sample = self.generate_sample(sample_idx=i + 1, topology=topology, split=split)
            samples.append(sample)

        return samples

    @staticmethod
    def save_dataset_jsonl(samples: list[dict[str, Any]], filepath: str | Path) -> None:
        """Serialize sample records to JSON Lines format with UTF-8 encoding."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def generate_and_save(
        self,
        filepath: str | Path,
        num_samples: int = 1000,
        split: str = "in_distribution",
    ) -> list[dict[str, Any]]:
        """Generate and write a complete dataset artifact."""
        samples = self.generate_dataset(num_samples=num_samples, split=split)
        self.save_dataset_jsonl(samples, filepath)
        return samples
