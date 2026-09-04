"""
topology.py — service dependency graph + blast-radius scoring.

Phase: 1 (baseline environment)

This module builds a directed graph of "who calls who" between microservices
and computes a blast-radius score per service: roughly, "if this service goes
down, how much of the rest of the app is affected?"

v1 uses a simple, explainable metric (weighted ancestor count / PageRank),
NOT a full GNN. That's intentional — Phase 2.5 can swap this for a learned
graph embedding without touching the rest of the codebase, as long as it
still exposes `blast_radius(service_id) -> float in [0, 1]`.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


class ServiceTopology:
    """
    Directed graph: edge (A -> B) means "A calls B" (A depends on B).
    Blast radius of B = how many services would notice if B breaks,
    i.e. how many nodes can reach B (B's ancestors / callers, transitively).
    """

    def __init__(self, n_services: int, edge_prob: float = 0.25, seed: int | None = None):
        self.n_services = n_services
        self.rng = np.random.default_rng(seed)
        self.graph = self._generate_graph(n_services, edge_prob)
        self._blast_radius_cache = self._compute_blast_radius()

    def _generate_graph(self, n: int, edge_prob: float) -> nx.DiGraph:
        """
        Generate a random-but-realistic microservice call graph:
        a DAG-ish structure with a few "hub" services (like auth, db-gateway)
        that many others depend on, plus some leaf services (like logging).
        """
        g = nx.gnp_random_graph(n, edge_prob, seed=int(self.rng.integers(1e9)), directed=True)
        g = nx.DiGraph([(u, v) for u, v in g.edges() if u != v])
        g.add_nodes_from(range(n))

        # Force a few designated "hub" services to have many dependents,
        # so blast-radius scores aren't flat/uninteresting.
        n_hubs = max(1, n // 5)
        hubs = self.rng.choice(n, size=n_hubs, replace=False)
        for hub in hubs:
            dependents = self.rng.choice(
                [i for i in range(n) if i != hub],
                size=min(n - 1, max(2, n // 2)),
                replace=False,
            )
            for dep in dependents:
                g.add_edge(int(dep), int(hub))  # dependent -> hub (dependent calls hub)

        # Ensure the graph is acyclic-ish for interpretability; break cycles if any.
        while not nx.is_directed_acyclic_graph(g):
            try:
                cycle = nx.find_cycle(g)
                g.remove_edge(*cycle[0])
            except nx.NetworkXNoCycle:
                break

        return g

    def _compute_blast_radius(self) -> dict[int, float]:
        """
        blast_radius[s] = (# services that transitively depend on s) / (n-1),
        clipped to [0, 1]. A pure leaf (nobody depends on it) -> ~0.
        A central hub (everyone depends on it) -> ~1.
        """
        scores: dict[int, float] = {}
        n = self.n_services
        for s in self.graph.nodes:
            ancestors = nx.ancestors(self.graph, s)  # nodes that can reach s
            scores[s] = len(ancestors) / max(1, n - 1)
        return scores

    def blast_radius(self, service_id: int) -> float:
        return float(self._blast_radius_cache.get(service_id, 0.0))

    def blast_radius_vector(self) -> np.ndarray:
        return np.array([self.blast_radius(i) for i in range(self.n_services)], dtype=np.float32)

    def dependents(self, service_id: int) -> set[int]:
        """Direct callers of this service (would feel an immediate hit)."""
        return set(self.graph.predecessors(service_id))

    def dependencies(self, service_id: int) -> set[int]:
        """Services this one calls (a failure here can cascade upstream to us)."""
        return set(self.graph.successors(service_id))

    def summary(self) -> str:
        lines = [f"ServiceTopology(n={self.n_services}, edges={self.graph.number_of_edges()})"]
        ranked = sorted(self._blast_radius_cache.items(), key=lambda kv: -kv[1])
        for sid, score in ranked:
            lines.append(f"  service_{sid:02d}: blast_radius={score:.2f}")
        return "\n".join(lines)


if __name__ == "__main__":
    topo = ServiceTopology(n_services=8, seed=42)
    print(topo.summary())
