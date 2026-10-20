"""
Strict directional relations and foundational data structures.

Formalizes relational logic over a strict, transitive, asymmetric directional
relation R (strictly 'left_of') on distinct entities.

EPISTEMIC NOTE:
Relational queries in CrossModalLab are evaluated strictly from the supplied premises
(representing a partial multimodal evidentiary state), NOT from an assumed fully observed
physical ground scene. An underdetermined state reflects insufficient evidence within
the premises, rather than factual falsehood or negation.
"""

from dataclasses import dataclass
from enum import Enum


class DirectionalRelation(str, Enum):
    """The directional relation. Strictly LEFT_OF ('left_of')."""
    LEFT_OF = "left_of"


class Modality(str, Enum):
    """The presentation modality for a premise."""
    VISUAL = "visual"
    TEXTUAL = "textual"


class EpistemicStatus(str, Enum):
    """
    Epistemic truth status of a query given a premise set.

    ENTAILED: Query relation holds in 100% of consistent linear extensions.
    CONTRADICTED: Reverse of query relation holds in 100% of consistent linear extensions.
    UNDERDETERMINED: Query relation holds in some extensions and reverse in others (insufficient evidence).
    INCONSISTENT: Premise set contains a logical contradiction or cycle (0 valid extensions).
    """
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    UNDERDETERMINED = "underdetermined"
    INCONSISTENT = "inconsistent"


class InconsistentPremisesError(ValueError):
    """Raised when premises are mutually contradictory (e.g., cycles or asymmetry violations)."""
    pass


class InvalidRelationError(ValueError):
    """Raised when a relation or entity specification violates formal axioms."""
    pass


@dataclass(frozen=True)
class Premise:
    """
    An asserted directional relationship between two distinct entities.

    Axioms enforced:
    - Irreflexivity: subject != object
    - Non-empty entity names
    - Strictly DirectionalRelation.LEFT_OF
    - Modality must be VISUAL or TEXTUAL
    """
    subject: str
    object: str
    relation: DirectionalRelation = DirectionalRelation.LEFT_OF
    modality: Modality = Modality.TEXTUAL

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise InvalidRelationError(f"Subject must be a non-empty string, got {self.subject!r}")
        if not isinstance(self.object, str) or not self.object.strip():
            raise InvalidRelationError(f"Object must be a non-empty string, got {self.object!r}")
        if self.subject.strip() == self.object.strip():
            raise InvalidRelationError(
                f"Irreflexivity violation: subject and object cannot be identical ({self.subject!r})"
            )

        # Normalize/validate relation
        if isinstance(self.relation, str):
            try:
                rel = DirectionalRelation(self.relation)
                object.__setattr__(self, "relation", rel)
            except ValueError:
                raise InvalidRelationError(
                    f"Unsupported relation: {self.relation!r}. Only 'left_of' is supported."
                )
        elif self.relation != DirectionalRelation.LEFT_OF:
            raise InvalidRelationError(
                f"Unsupported relation: {self.relation!r}. Only DirectionalRelation.LEFT_OF is supported."
            )

        # Normalize/validate modality
        if isinstance(self.modality, str):
            try:
                mod = Modality(self.modality)
                object.__setattr__(self, "modality", mod)
            except ValueError:
                raise InvalidRelationError(
                    f"Unsupported modality: {self.modality!r}. Must be 'visual' or 'textual'."
                )
        elif not isinstance(self.modality, Modality):
            raise InvalidRelationError(
                f"Unsupported modality: {self.modality!r}. Must be an instance of Modality."
            )


@dataclass(frozen=True)
class Query:
    """
    A target relational query between two distinct entities.

    Evaluates whether R(subject, object) is entailed, contradicted, or underdetermined.
    """
    subject: str
    object: str
    relation: DirectionalRelation = DirectionalRelation.LEFT_OF

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise InvalidRelationError(f"Subject must be a non-empty string, got {self.subject!r}")
        if not isinstance(self.object, str) or not self.object.strip():
            raise InvalidRelationError(f"Object must be a non-empty string, got {self.object!r}")
        if self.subject.strip() == self.object.strip():
            raise InvalidRelationError(
                f"Irreflexivity violation: query subject and object cannot be identical ({self.subject!r})"
            )

        if isinstance(self.relation, str):
            try:
                rel = DirectionalRelation(self.relation)
                object.__setattr__(self, "relation", rel)
            except ValueError:
                raise InvalidRelationError(
                    f"Unsupported relation: {self.relation!r}. Only 'left_of' is supported."
                )
        elif self.relation != DirectionalRelation.LEFT_OF:
            raise InvalidRelationError(
                f"Unsupported relation: {self.relation!r}. Only DirectionalRelation.LEFT_OF is supported."
            )
