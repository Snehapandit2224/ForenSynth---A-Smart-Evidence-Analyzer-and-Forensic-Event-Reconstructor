"""Graph construction service for entity resolution.

Stage 7: Builds observation graph from classified edges.

Key Principle:
- ONLY confirmed edges ([>= 0.80]) are added to graph
- Candidate and rejected edges are ignored (not merged into clusters)
- Nodes = observation IDs
- Edges = confirmed edges connecting observations of same entity

Strategy:
- High-confidence decisions only drive clustering
- Candidate edges allow manual review without forcing merges
- Graph ready for connected components analysis (Stage 8)

Output: NetworkX Graph + statistics
"""

from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
import networkx as nx

from .edge_classifier import ClassifiedEdge, EdgeClassification


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class GraphNode:
    """Node in observation graph."""

    obs_id: str
    degree: int = 0  # Number of connected edges

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "obs_id": self.obs_id,
            "degree": self.degree,
        }


@dataclass
class GraphEdge:
    """Edge in observation graph (confirmed edge only)."""

    obs_id_1: str
    obs_id_2: str
    similarity_score: float  # For reference/auditing

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "obs_id_1": self.obs_id_1,
            "obs_id_2": self.obs_id_2,
            "similarity_score": round(self.similarity_score, 3),
        }


@dataclass
class GraphBuildingReport:
    """Report from graph construction stage."""

    total_observations: int = 0  # Unique obs_ids
    total_confirmed_edges: int = 0
    total_candidate_edges: int = 0  # Not added to graph (for reference)
    total_rejected_edges: int = 0  # Not added to graph (for reference)
    
    # Graph statistics
    nodes_count: int = 0
    edges_count: int = 0
    
    # Node degree statistics
    avg_node_degree: float = 0.0
    max_node_degree: int = 0
    min_node_degree: int = 0
    isolated_nodes: int = 0  # Degree 0 (no confirmed edges)
    
    # Edge statistics
    avg_edge_similarity: float = 0.0
    min_edge_similarity: float = 1.0
    max_edge_similarity: float = 0.0
    
    processing_time_sec: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "total_observations": self.total_observations,
            "confirmed_edges": self.total_confirmed_edges,
            "candidate_edges": self.total_candidate_edges,
            "rejected_edges": self.total_rejected_edges,
            "nodes": self.nodes_count,
            "edges": self.edges_count,
            "avg_node_degree": round(self.avg_node_degree, 2),
            "max_node_degree": self.max_node_degree,
            "min_node_degree": self.min_node_degree,
            "isolated_nodes": self.isolated_nodes,
            "avg_edge_similarity": round(self.avg_edge_similarity, 3),
            "min_edge_similarity": round(self.min_edge_similarity, 3),
            "max_edge_similarity": round(self.max_edge_similarity, 3),
            "processing_time_sec": round(self.processing_time_sec, 3),
        }


# ============================================================================
# Graph Building Implementation
# ============================================================================


class GraphBuilder:
    """Constructs observation graph from classified edges."""

    def __init__(self):
        """Initialize graph builder."""
        pass

    def build_graph(
        self, classified_edges: List[ClassifiedEdge], all_obs_ids: List[str]
    ) -> Tuple[nx.Graph, GraphBuildingReport]:
        """
        Build observation graph from classified edges.

        Strategy:
        - Create graph with all observations as nodes
        - Add edges ONLY for confirmed classifications
        - Candidate/rejected edges are ignored
        - Allows manual review of borderline pairs without forcing merges

        Args:
            classified_edges: List of ClassifiedEdge from Stage 6
            all_obs_ids: List of all observation IDs in case

        Returns:
            Tuple of (graph, report)
            - graph: NetworkX undirected Graph with nodes and edges
            - report: GraphBuildingReport with statistics
        """
        import time
        start_time = time.time()

        # Create graph
        graph = nx.Graph()

        # Add all observations as nodes
        for obs_id in all_obs_ids:
            graph.add_node(obs_id)

        # Track edges by classification
        confirmed_edges: List[GraphEdge] = []
        candidate_count = 0
        rejected_count = 0
        edge_similarities: List[float] = []

        # Add only CONFIRMED edges to graph
        for edge in classified_edges:
            if edge.classification == EdgeClassification.CONFIRMED:
                # Add edge to graph
                graph.add_edge(edge.obs_id_1, edge.obs_id_2, weight=edge.similarity_score)
                
                graph_edge = GraphEdge(
                    obs_id_1=edge.obs_id_1,
                    obs_id_2=edge.obs_id_2,
                    similarity_score=edge.similarity_score,
                )
                confirmed_edges.append(graph_edge)
                edge_similarities.append(edge.similarity_score)
            elif edge.classification == EdgeClassification.CANDIDATE:
                candidate_count += 1
            else:  # REJECTED
                rejected_count += 1

        # Compute node degree statistics
        degrees = [graph.degree(node) for node in graph.nodes()]
        isolated = sum(1 for d in degrees if d == 0)

        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0

        # Compute edge statistics
        avg_similarity = sum(edge_similarities) / len(edge_similarities) if edge_similarities else 0.0
        min_similarity = min(edge_similarities) if edge_similarities else 1.0
        max_similarity = max(edge_similarities) if edge_similarities else 0.0

        elapsed = time.time() - start_time

        report = GraphBuildingReport(
            total_observations=len(all_obs_ids),
            total_confirmed_edges=len(confirmed_edges),
            total_candidate_edges=candidate_count,
            total_rejected_edges=rejected_count,
            nodes_count=graph.number_of_nodes(),
            edges_count=graph.number_of_edges(),
            avg_node_degree=avg_degree,
            max_node_degree=max_degree,
            min_node_degree=min_degree,
            isolated_nodes=isolated,
            avg_edge_similarity=avg_similarity,
            min_edge_similarity=min_similarity,
            max_edge_similarity=max_similarity,
            processing_time_sec=elapsed,
        )

        return graph, report
