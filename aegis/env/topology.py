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

    Supports multiple criticality definitions (`criticality_mode`):
      - "ancestors": unweighted fraction of transitive callers |Ancestors(v)| / (|V|-1)
      - "traffic_weighted": callers weighted by synthetic per-service request traffic weights
      - "pagerank": reverse-graph PageRank centrality normalized to [0, 1]
    """

    def __init__(
        self,
        n_services: int,
        edge_prob: float = 0.25,
        seed: int | None = None,
        criticality_mode: str = "ancestors",
    ):
        self.n_services = n_services
        self.criticality_mode = criticality_mode
        self.rng = np.random.default_rng(seed)
        self.graph = self._generate_graph(n_services, edge_prob)
        # Assign synthetic request traffic weights in [0.5, 2.5] per service
        self.traffic_weights = {
            i: float(self.rng.uniform(0.5, 2.5)) for i in range(n_services)
        }
        self._blast_radius_cache = self._compute_blast_radius(mode=criticality_mode)
        self._ancestor_blast_radius_cache = self._compute_blast_radius(mode="ancestors")

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

    def _compute_blast_radius(self, mode: str = "ancestors") -> dict[int, float]:
        """
        Compute normalized criticality / blast-radius score in [0, 1] per service.
        - "ancestors": (# services that transitively depend on s) / (n-1).
        - "traffic_weighted": traffic-weighted fraction of transitive callers.
        - "pagerank": PageRank centrality on the dependency graph normalized to [0, 1].
        """
        scores: dict[int, float] = {}
        n = self.n_services
        if mode == "traffic_weighted":
            total_traffic = sum(self.traffic_weights.values())
            for s in self.graph.nodes:
                ancestors = nx.ancestors(self.graph, s)
                other_traffic = max(1e-6, total_traffic - self.traffic_weights[s])
                scores[s] = float(
                    np.clip(
                        sum(self.traffic_weights[a] for a in ancestors) / other_traffic,
                        0.0,
                        1.0,
                    )
                )
            return scores
        if mode == "pagerank":
            from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_numpy

            pr = _pagerank_numpy(self.graph, alpha=0.85)
            pr_vals = list(pr.values())
            lo, hi = min(pr_vals), max(pr_vals)
            span = max(1e-6, hi - lo)
            for s in self.graph.nodes:
                scores[s] = float(np.clip((pr[s] - lo) / span, 0.0, 1.0))
            return scores

        # Default: unweighted ancestor fraction
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
