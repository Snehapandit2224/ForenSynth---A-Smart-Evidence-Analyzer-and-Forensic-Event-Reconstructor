"""Clustering service for entity resolution.

Stage 8: Performs entity clustering using connected components.

Strategy:
- Each connected component in the graph represents a canonical entity
- Nodes in same component = observations of same entity
- Using NetworkX's connected_components() for efficiency
- Returns clusters as lists of observation IDs

Principle:
- Only CONFIRMED edges merge clusters (from Stage 7)
- Candidate edges already filtered out - no borderline merges
- Clear, deterministic clustering based on high-confidence decisions

Output: List of clusters + statistics
"""

from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field
import networkx as nx


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class EntityCluster:
    """A cluster representing a canonical entity."""

    cluster_id: str  # e.g., "C1", "C2", etc.
    obs_ids: List[str]  # Observation IDs in this cluster
    size: int = field(init=False)  # Number of observations
    
    def __post_init__(self):
        """Compute size after initialization."""
        self.size = len(self.obs_ids)

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "obs_ids": sorted(self.obs_ids),
        }


@dataclass
class ClusteringReport:
    """Report from clustering stage."""

    total_observations: int = 0
    total_clusters: int = 0
    
    # Cluster size statistics
    avg_cluster_size: float = 0.0
    max_cluster_size: int = 0
    min_cluster_size: int = 0
    
    # Size distribution
    singleton_clusters: int = 0  # Size 1 (isolated observations)
    pair_clusters: int = 0  # Size 2
    triplet_clusters: int = 0  # Size 3
    large_clusters: int = 0  # Size >= 4
    
    # Coverage
    clustered_coverage_ratio: float = 0.0  # (Total observations) / total_observations
    
    processing_time_sec: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "total_observations": self.total_observations,
            "total_clusters": self.total_clusters,
            "avg_cluster_size": round(self.avg_cluster_size, 2),
            "max_cluster_size": self.max_cluster_size,
            "min_cluster_size": self.min_cluster_size,
            "singleton_clusters": self.singleton_clusters,
            "pair_clusters": self.pair_clusters,
            "triplet_clusters": self.triplet_clusters,
            "large_clusters": self.large_clusters,
            "coverage_ratio": round(self.clustered_coverage_ratio, 3),
            "processing_time_sec": round(self.processing_time_sec, 3),
        }


# ============================================================================
# Clustering Implementation
# ============================================================================


class Clusterer:
    """Performs entity clustering on observation graph."""

    def __init__(self):
        """Initialize clusterer."""
        pass

    def cluster_observations(
        self, graph: nx.Graph, all_obs_ids: List[str]
    ) -> Tuple[List[EntityCluster], ClusteringReport]:
        """
        Cluster observations using connected components.

        Strategy:
        - Each connected component is one canonical entity
        - Observations in same component are highly likely to be same entity (confirmed edges only)
        - Isolated nodes (no confirmed edges) form singleton clusters

        Args:
            graph: NetworkX graph from Stage 7
            all_obs_ids: List of all observation IDs

        Returns:
            Tuple of (clusters, report)
            - clusters: List of EntityCluster, sorted by size descending
            - report: ClusteringReport with statistics
        """
        import time
        start_time = time.time()

        # Extract connected components
        components = list(nx.connected_components(graph))

        # Create clusters
        clusters: List[EntityCluster] = []
        for idx, component in enumerate(components):
            cluster_id = f"C{idx + 1}"
            obs_ids = sorted(list(component))
            
            cluster = EntityCluster(
                cluster_id=cluster_id,
                obs_ids=obs_ids,
            )
            clusters.append(cluster)

        # Sort clusters by size (descending) for presentation
        clusters.sort(key=lambda c: c.size, reverse=True)

        # Compute statistics
        sizes = [c.size for c in clusters]
        singletons = sum(1 for s in sizes if s == 1)
        pairs = sum(1 for s in sizes if s == 2)
        triplets = sum(1 for s in sizes if s == 3)
        large = sum(1 for s in sizes if s >= 4)

        avg_size = sum(sizes) / len(sizes) if sizes else 0.0
        max_size = max(sizes) if sizes else 0
        min_size = min(sizes) if sizes else 0

        # Coverage: should always be 1.0 (all obs_ids must be in exactly one cluster)
        total_in_clusters = sum(c.size for c in clusters)
        coverage = total_in_clusters / len(all_obs_ids) if all_obs_ids else 0.0

        elapsed = time.time() - start_time

        report = ClusteringReport(
            total_observations=len(all_obs_ids),
            total_clusters=len(clusters),
            avg_cluster_size=avg_size,
            max_cluster_size=max_size,
            min_cluster_size=min_size,
            singleton_clusters=singletons,
            pair_clusters=pairs,
            triplet_clusters=triplets,
            large_clusters=large,
            clustered_coverage_ratio=coverage,
            processing_time_sec=elapsed,
        )

        return clusters, report

    def get_cluster_for_obs(self, clusters: List[EntityCluster], obs_id: str) -> Tuple[EntityCluster, int]:
        """
        Find the cluster containing a specific observation.

        Args:
            clusters: List of EntityCluster
            obs_id: Observation ID to find

        Returns:
            Tuple of (cluster, position_in_cluster) or (None, -1) if not found
        """
        for cluster in clusters:
            if obs_id in cluster.obs_ids:
                position = cluster.obs_ids.index(obs_id)
                return cluster, position
        
        return None, -1

    def get_cluster_relationships(
        self, clusters: List[EntityCluster]
    ) -> Dict[str, List[str]]:
        """
        Identify clusters with only 1 observation (singletons).

        Args:
            clusters: List of EntityCluster

        Returns:
            Dict mapping singleton obs_ids to their singleton cluster IDs
        """
        singletons = {}
        for cluster in clusters:
            if cluster.size == 1:
                obs_id = cluster.obs_ids[0]
                singletons[obs_id] = cluster.cluster_id
        
        return singletons
