"""
Unit tests for strict directional relations and dataclass axiom enforcement.
"""

from dataclasses import FrozenInstanceError
import unittest

from src.symbolic.relations import (
    DirectionalRelation,
    EpistemicStatus,
    InvalidRelationError,
    Modality,
    Premise,
    Query,
)


class TestRelations(unittest.TestCase):
    """Test suite for relations, premises, queries, and logical axioms."""

    def test_valid_premise_creation(self) -> None:
        p = Premise(
            subject="blue_cube",
            object="red_sphere",
            relation=DirectionalRelation.LEFT_OF,
            modality=Modality.VISUAL,
        )
        self.assertEqual(p.subject, "blue_cube")
        self.assertEqual(p.object, "red_sphere")
        self.assertEqual(p.relation, DirectionalRelation.LEFT_OF)
        self.assertEqual(p.modality, Modality.VISUAL)

    def test_premise_string_coercion(self) -> None:
        p = Premise(
            subject="A",
            object="B",
            relation="left_of",  # string coerced to enum
            modality="textual",   # string coerced to enum
        )
        self.assertEqual(p.relation, DirectionalRelation.LEFT_OF)
        self.assertEqual(p.modality, Modality.TEXTUAL)

    def test_premise_irreflexivity(self) -> None:
        """Axiom: An entity cannot be left of itself."""
        with self.assertRaises(InvalidRelationError) as ctx:
            Premise(subject="blue_cube", object="blue_cube")
        self.assertIn("Irreflexivity violation", str(ctx.exception))

    def test_premise_empty_subject_or_object(self) -> None:
        with self.assertRaises(InvalidRelationError):
            Premise(subject="", object="red_sphere")
        with self.assertRaises(InvalidRelationError):
            Premise(subject="blue_cube", object="   ")

    def test_premise_unsupported_relation(self) -> None:
        with self.assertRaises(InvalidRelationError):
            Premise(subject="A", object="B", relation="right_of")  # type: ignore

    def test_premise_unsupported_modality(self) -> None:
        with self.assertRaises(InvalidRelationError):
            Premise(subject="A", object="B", modality="audio")  # type: ignore

    def test_premise_immutability(self) -> None:
        p = Premise(subject="A", object="B")
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            p.subject = "C"  # type: ignore

    def test_valid_query_creation(self) -> None:
        q = Query(subject="A", object="C", relation=DirectionalRelation.LEFT_OF)
        self.assertEqual(q.subject, "A")
        self.assertEqual(q.object, "C")
        self.assertEqual(q.relation, DirectionalRelation.LEFT_OF)

    def test_query_irreflexivity(self) -> None:
        with self.assertRaises(InvalidRelationError) as ctx:
            Query(subject="entity_x", object="entity_x")
        self.assertIn("Irreflexivity violation", str(ctx.exception))

    def test_query_empty_subject_or_object(self) -> None:
        with self.assertRaises(InvalidRelationError):
            Query(subject="", object="C")
        with self.assertRaises(InvalidRelationError):
            Query(subject="A", object="")

    def test_query_immutability(self) -> None:
        q = Query(subject="A", object="B")
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            q.subject = "Z"  # type: ignore

    def test_epistemic_status_values(self) -> None:
        self.assertEqual(EpistemicStatus.ENTAILED.value, "entailed")
        self.assertEqual(EpistemicStatus.CONTRADICTED.value, "contradicted")
        self.assertEqual(EpistemicStatus.UNDERDETERMINED.value, "underdetermined")
        self.assertEqual(EpistemicStatus.INCONSISTENT.value, "inconsistent")


if __name__ == "__main__":
    unittest.main()
