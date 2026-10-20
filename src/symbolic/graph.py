"""
Graph representations, cycle detection, reachability, and linear extensions.

Implements directed acyclic graph (DAG) modeling of partial orders over
the strict directional relation R ('left_of').
"""

from collections import deque
from typing import Optional

from src.symbolic.relations import (
    InconsistentPremisesError,
    InvalidRelationError,
    Premise,
)


class RelationGraph:
    """
    A directed graph representing partial orders over strict directional relations.

    An edge u -> v denotes that entity u is strictly to the left of entity v (u < v).
    """

    def __init__(self, entities: Optional[list[str]] = None) -> None:
        self._entities: set[str] = set()
        self._adj: dict[str, set[str]] = {}
        self._pred: dict[str, set[str]] = {}

        if entities:
            for entity in entities:
                self.add_entity(entity)

    @property
    def entities(self) -> set[str]:
        """Set of all registered entities in the graph."""
        return set(self._entities)

    @property
    def edges(self) -> set[tuple[str, str]]:
        """Set of all directed edges (u, v) in the graph."""
        edge_set: set[tuple[str, str]] = set()
        for u, neighbors in self._adj.items():
            for v in neighbors:
                edge_set.add((u, v))
        return edge_set

    def add_entity(self, entity: str) -> None:
        """Register an entity node in the graph."""
        if not isinstance(entity, str) or not entity.strip():
            raise InvalidRelationError(f"Entity name must be a non-empty string, got {entity!r}")
        name = entity.strip()
        if name not in self._entities:
            self._entities.add(name)
            self._adj[name] = set()
            self._pred[name] = set()

    def add_relation(self, u: str, v: str, check_cycle: bool = True) -> None:
        """
        Add directed relation u -> v (u is left_of v).

        Args:
            u: Subject entity (left).
            v: Object entity (right).
            check_cycle: If True, validates irreflexivity, asymmetry, and acyclicity.

        Raises:
            InvalidRelationError: If u == v.
            InconsistentPremisesError: If adding u -> v creates a 2-cycle (asymmetry violation)
                                       or an n-cycle (transitive cycle violation).
        """
        if not isinstance(u, str) or not u.strip():
            raise InvalidRelationError(f"Subject must be a non-empty string, got {u!r}")
        if not isinstance(v, str) or not v.strip():
            raise InvalidRelationError(f"Object must be a non-empty string, got {v!r}")

        u_norm = u.strip()
        v_norm = v.strip()

        if u_norm == v_norm:
            raise InvalidRelationError(
                f"Irreflexivity violation: cannot relate entity {u_norm!r} to itself"
            )

        self.add_entity(u_norm)
        self.add_entity(v_norm)

        # Idempotent: already present
        if v_norm in self._adj[u_norm]:
            return

        if check_cycle:
            # Check 1: Asymmetry (immediate 2-cycle)
            if u_norm in self._adj[v_norm]:
                raise InconsistentPremisesError(
                    f"Asymmetry violation: {v_norm} -> {u_norm} is already asserted; "
                    f"cannot assert {u_norm} -> {v_norm}"
                )

            # Check 2: Cycle detection via reachability (if v can reach u, then u -> v creates a cycle)
            if self.is_reachable(v_norm, u_norm):
                raise InconsistentPremisesError(
                    f"Cycle detected: {u_norm} is already reachable from {v_norm}; "
                    f"adding {u_norm} -> {v_norm} creates a directed cycle"
                )

        self._adj[u_norm].add(v_norm)
        self._pred[v_norm].add(u_norm)

    @classmethod
    def from_premises(
        cls,
        premises: list[Premise],
        check_cycle: bool = True,
        extra_entities: Optional[list[str]] = None,
    ) -> "RelationGraph":
        """Construct a RelationGraph from a sequence of Premise objects."""
        graph = cls(entities=extra_entities)
        for p in premises:
            graph.add_relation(p.subject, p.object, check_cycle=check_cycle)
        return graph

    def is_reachable(self, source: str, target: str) -> bool:
        """
        Check if there is a directed path from source to target.

        Uses breadth-first search (BFS).
        """
        source_norm = source.strip()
        target_norm = target.strip()

        if source_norm not in self._entities or target_norm not in self._entities:
            return False

        if source_norm == target_norm:
            return True

        visited: set[str] = {source_norm}
        queue: deque[str] = deque([source_norm])

        while queue:
            curr = queue.popleft()
            for neighbor in self._adj.get(curr, ()):
                if neighbor == target_norm:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def has_cycle(self) -> bool:
        """
        Detect whether the graph contains any directed cycle using 3-color DFS.

        Returns:
            True if at least one cycle exists, False otherwise.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {node: WHITE for node in self._entities}

        def _dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in self._adj.get(node, ()):
                if color[neighbor] == GRAY:
                    return True
                if color[neighbor] == WHITE:
                    if _dfs(neighbor):
                        return True
            color[node] = BLACK
            return False

        for node in sorted(self._entities):
            if color[node] == WHITE:
                if _dfs(node):
                    return True

        return False

    def find_cycles(self) -> list[list[str]]:
        """
        Find and return any directed cycles in the graph for diagnostic reporting.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {node: WHITE for node in self._entities}
        parent: dict[str, Optional[str]] = {node: None for node in self._entities}
        cycles: list[list[str]] = []

        def _dfs(node: str, path: list[str]) -> None:
            color[node] = GRAY
            path.append(node)

            for neighbor in sorted(self._adj.get(node, ())):
                if color[neighbor] == GRAY:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle_path = path[cycle_start:] + [neighbor]
                    cycles.append(cycle_path)
                elif color[neighbor] == WHITE:
                    parent[neighbor] = node
                    _dfs(neighbor, path)

            path.pop()
            color[node] = BLACK

        for node in sorted(self._entities):
            if color[node] == WHITE:
                _dfs(node, [])

        return cycles

    def transitive_closure(self) -> set[tuple[str, str]]:
        """
        Compute the transitive closure R^+ of all directed reachability pairs.

        Returns:
            Set of pairs (u, v) where u strictly precedes v in the partial order.
        """
        closure: set[tuple[str, str]] = set()
        for u in self._entities:
            for v in self._entities:
                if u != v and self.is_reachable(u, v):
                    closure.add((u, v))
        return closure

    def get_linear_extensions(self) -> list[list[str]]:
        """
        Enumerate all valid linear extensions (topological sortings) of the graph entities.

        If the graph contains any cycle, returns an empty list [].

        Uses recursive backtracking over zero-in-degree nodes, producing
        deterministic, alphabetically ordered extensions.
        """
        if self.has_cycle():
            return []

        all_nodes = sorted(self._entities)
        if not all_nodes:
            return []

        # Compute initial in-degrees
        in_degree: dict[str, int] = {node: len(self._pred[node]) for node in all_nodes}
        visited: set[str] = set()
        results: list[list[str]] = []
        current_path: list[str] = []

        def _backtrack() -> None:
            if len(current_path) == len(all_nodes):
                results.append(list(current_path))
                return

            # Find all unvisited nodes with zero in-degree
            candidates = [
                node for node in all_nodes
                if node not in visited and in_degree[node] == 0
            ]

            for node in candidates:
                visited.add(node)
                current_path.append(node)

                # Decrement in-degree of all successors
                for successor in self._adj.get(node, ()):
                    in_degree[successor] -= 1

                _backtrack()

                # Restore
                for successor in self._adj.get(node, ()):
                    in_degree[successor] += 1

                current_path.pop()
                visited.remove(node)

        _backtrack()
        return results

    def validate_consistency(self) -> None:
        """
        Verify that the graph is strictly acyclic.

        Raises:
            InconsistentPremisesError: If any cycle is detected.
        """
        if self.has_cycle():
            cycles = self.find_cycles()
            cycle_desc = " -> ".join(cycles[0]) if cycles else "detected"
            raise InconsistentPremisesError(
                f"Premises contain a contradictory cycle: {cycle_desc}"
            )
