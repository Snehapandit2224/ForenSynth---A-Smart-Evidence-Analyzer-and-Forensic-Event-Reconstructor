"""
ForenSynth – Timeline Agent
graph_builder.py: builds a directed NetworkX graph from events and edges,
and serialises it for the Critique Agent / Visualisation Layer.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import networkx as nx

from models import TimelineEdge, TimelineEvent

log = logging.getLogger("forensynth.timeline.graph_builder")


class GraphBuilder:
    """
    Stage 4 / Stage 13 – Event Graph Construction and Export.

    Node attributes:
        event_id, timestamp, entity_id, primary_alias, confidence, location, modality, content

    Edge attributes:
        edge_type, confidence, relation, label
    """

    def build(
        self,
        events: List[TimelineEvent],
        temporal_edges: List[TimelineEdge],
        causal_edges: List[TimelineEdge],
    ) -> nx.DiGraph:
        G = nx.DiGraph()

        # ── Nodes ─────────────────────────────────────────────────────────────
        for ev in events:
            G.add_node(
                ev.event_id,
                timestamp=ev.timestamp,
                ts_epoch=ev.ts_epoch,
                entity_id=ev.entity_id,
                primary_alias=ev.primary_alias,
                confidence=round(ev.confidence, 4),
                location=ev.location,
                modality=ev.modality,
                content=ev.content[:120],   # truncate for graph readability
                conflict_flag=ev.conflict_flag,
                obs_ids=ev.obs_ids,
            )

        # ── Edges ─────────────────────────────────────────────────────────────
        for edge in [*temporal_edges, *causal_edges]:
            if not G.has_node(edge.source) or not G.has_node(edge.target):
                log.debug("Edge references missing node (%s → %s); skipping", edge.source, edge.target)
                continue
            # If a causal edge already exists for this pair, prefer CAUSAL over TEMPORAL
            if G.has_edge(edge.source, edge.target):
                existing = G[edge.source][edge.target]
                if existing.get("edge_type") == "CAUSAL" and edge.edge_type.value == "TEMPORAL":
                    continue   # keep the richer causal edge
            G.add_edge(
                edge.source,
                edge.target,
                edge_type=edge.edge_type.value,
                confidence=round(edge.confidence, 4),
                relation=edge.relation.value,
                label=edge.label,
            )

        log.info(
            "Graph built: %d nodes, %d edges",
            G.number_of_nodes(),
            G.number_of_edges(),
        )
        return G

    def to_export_dict(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Serialise the graph to a plain dict for JSON export.
        Format is compatible with the Critique Agent and Visualisation Layer.
        """
        nodes = []
        for node_id, attrs in G.nodes(data=True):
            nodes.append({"id": node_id, **attrs})

        edges = []
        for src, tgt, attrs in G.edges(data=True):
            edges.append({"source": src, "target": tgt, **attrs})

        # Compute causal_links subset (for top-level output field)
        causal_links = [e for e in edges if e.get("edge_type") == "CAUSAL"]

        return {
            "nodes":        nodes,
            "edges":        edges,
            "causal_links": causal_links,
            "node_count":   G.number_of_nodes(),
            "edge_count":   G.number_of_edges(),
        }

    def export_to_file(self, G: nx.DiGraph, path: str) -> None:
        """Write timeline_graph.json to disk."""
        export = self.to_export_dict(G)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(export, fh, indent=2, ensure_ascii=False)
        log.info("Graph exported → %s", path)
