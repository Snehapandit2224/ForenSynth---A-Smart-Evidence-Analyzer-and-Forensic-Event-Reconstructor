"""Conflict handler service for entity resolution.

Stage 10: Detects and conservatively resolves conflicts.

Strategy:
- Identify conflicts without forcing resolutions
- Conservative approach: flag for review rather than auto-merge
- Detect contradictions (role conflicts, temporal conflicts)
- Provide recommendations without modifying cluster structure

Conflicts Detected:
1. Role contradictions (suspect + witness in same cluster)
2. Low-confidence clusters
3. Temporal conflicts (same person in two places at once)
4. Size anomalies (suspicious cluster sizes)

Output: List of conflicts + recommendations
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from .clusterer import EntityCluster


# ============================================================================
# Data Structures
# ============================================================================


class ConflictType(str, Enum):
    """Types of conflicts detected."""

    ROLE_CONTRADICTION = "role_contradiction"  # Incompatible roles in same cluster
    TEMPORAL_CONFLICT = "temporal_conflict"  # Same entity in two places at once
    LOW_CONFIDENCE = "low_confidence"  # Cluster built on weak links
    SUSPICIOUS_SIZE = "suspicious_size"  # Unusually large cluster
    ATTACHMENT_CONFLICT = "attachment_conflict"  # Conflict with attachment


class ConflictSeverity(str, Enum):
    """Severity levels for conflicts."""

    HIGH = "high"  # Likely merge error, needs review
    MEDIUM = "medium"  # Potential issue, recommend review
    LOW = "low"  # Minor inconsistency, may be acceptable


@dataclass
class Resolution:
    """Proposed resolution for a conflict."""

    action: str  # "merge", "split", "flag_for_review", "accept"
    rationale: str
    confidence: float  # [0.0, 1.0] confidence in recommendation


@dataclass
class DetectedConflict:
    """A detected conflict in entity resolution."""

    conflict_id: str  # e.g., "CFT_001"
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    
    # Context
    affected_clusters: List[str] = field(default_factory=list)  # Cluster IDs
    affected_observations: List[str] = field(default_factory=list)  # Obs IDs
    
    # Evidence
    evidence: List[str] = field(default_factory=list)  # Why this is a conflict
    
    # Resolution
    recommendation: str = ""
    conservative_action: str = ""  # What to do conservatively

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "conflict_id": self.conflict_id,
            "type": self.conflict_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "affected_clusters": self.affected_clusters,
            "affected_observations": self.affected_observations,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "conservative_action": self.conservative_action,
        }


@dataclass
class ConflictReport:
    """Report from conflict detection stage."""

    total_clusters: int = 0
    total_observations: int = 0
    total_conflicts_detected: int = 0
    
    # Conflict distribution
    high_severity_conflicts: int = 0
    medium_severity_conflicts: int = 0
    low_severity_conflicts: int = 0
    
    # Conflict types
    role_contradiction_count: int = 0
    temporal_conflict_count: int = 0
    low_confidence_count: int = 0
    suspicious_size_count: int = 0
    
    # Status
    clusters_with_conflicts: int = 0
    observations_affected: int = 0
    
    processing_time_sec: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "total_clusters": self.total_clusters,
            "total_observations": self.total_observations,
            "total_conflicts": self.total_conflicts_detected,
            "high_severity": self.high_severity_conflicts,
            "medium_severity": self.medium_severity_conflicts,
            "low_severity": self.low_severity_conflicts,
            "role_contradictions": self.role_contradiction_count,
            "temporal_conflicts": self.temporal_conflict_count,
            "low_confidence": self.low_confidence_count,
            "suspicious_sizes": self.suspicious_size_count,
            "clusters_affected": self.clusters_with_conflicts,
            "observations_affected": self.observations_affected,
            "processing_time_sec": round(self.processing_time_sec, 3),
        }


# ============================================================================
# Conflict Detection Implementation
# ============================================================================


class ConflictHandler:
    """Detects and analyzes conflicts conservatively."""

    def __init__(self):
        """Initialize conflict handler."""
        self._conflict_counter: int = 0

    def _detect_role_contradictions(
        self, clusters: List[EntityCluster], observations: List
    ) -> List[DetectedConflict]:
        """
        Detect role contradictions within clusters.

        Args:
            clusters: List of EntityCluster
            observations: List of NormalizedObservation

        Returns:
            List of DetectedConflict
        """
        conflicts = []
        obs_dict = {obs.obs_id: obs for obs in observations}

        # Role compatibility rules
        incompatible_roles = [
            ("suspect", "witness"),
            ("suspect", "victim"),
            ("witness", "perpetrator"),
        ]

        for cluster in clusters:
            roles = set()
            obs_by_role = {}

            for obs_id in cluster.obs_ids:
                obs = obs_dict.get(obs_id)
                if obs and hasattr(obs, 'role'):
                    role = obs.role
                    roles.add(role)
                    if role not in obs_by_role:
                        obs_by_role[role] = []
                    obs_by_role[role].append(obs_id)

            # Check for incompatibilities
            roles_list = sorted(list(roles))
            for role1, role2 in incompatible_roles:
                if role1 in roles and role2 in roles:
                    self._conflict_counter += 1
                    conflict = DetectedConflict(
                        conflict_id=f"CFT_{self._conflict_counter:03d}",
                        conflict_type=ConflictType.ROLE_CONTRADICTION,
                        severity=ConflictSeverity.HIGH,
                        description=f"Cluster {cluster.cluster_id} contains incompatible roles: {role1} and {role2}",
                        affected_clusters=[cluster.cluster_id],
                        affected_observations=cluster.obs_ids,
                        evidence=[
                            f"Observations with role '{role1}': {obs_by_role.get(role1, [])}",
                            f"Observations with role '{role2}': {obs_by_role.get(role2, [])}",
                        ],
                        recommendation="Consider splitting cluster into separate entities",
                        conservative_action="Flag for manual review - likely distinct entities",
                    )
                    conflicts.append(conflict)

        return conflicts

    def _detect_temporal_conflicts(
        self, clusters: List[EntityCluster], observations: List
    ) -> List[DetectedConflict]:
        """
        Detect temporal conflicts (same entity in two places at once).

        Args:
            clusters: List of EntityCluster
            observations: List of NormalizedObservation

        Returns:
            List of DetectedConflict
        """
        conflicts = []
        obs_dict = {obs.obs_id: obs for obs in observations}

        for cluster in clusters:
            obs_list = [obs_dict.get(obs_id) for obs_id in cluster.obs_ids]
            obs_list = [obs for obs in obs_list if obs is not None]

            # Check temporal sequence
            obs_list.sort(key=lambda o: o.timestamp_dt if hasattr(o, 'timestamp_dt') else o.timestamp)

            # Look for impossible temporal sequences
            for i in range(len(obs_list) - 1):
                curr = obs_list[i]
                next_obs = obs_list[i + 1]

                if hasattr(curr, 'timestamp_dt') and hasattr(next_obs, 'timestamp_dt'):
                    time_diff = (next_obs.timestamp_dt - curr.timestamp_dt).total_seconds()

                    # Same location - should have time gap
                    if (
                        time_diff < 5
                        and hasattr(curr, 'location')
                        and hasattr(next_obs, 'location')
                        and curr.location != next_obs.location
                    ):
                        # Different locations in less than 5 seconds - suspicious
                        self._conflict_counter += 1
                        conflict = DetectedConflict(
                            conflict_id=f"CFT_{self._conflict_counter:03d}",
                            conflict_type=ConflictType.TEMPORAL_CONFLICT,
                            severity=ConflictSeverity.MEDIUM,
                            description=(
                                f"Cluster {cluster.cluster_id}: same entity at different locations "
                                f"in {time_diff} seconds"
                            ),
                            affected_clusters=[cluster.cluster_id],
                            affected_observations=[curr.obs_id, next_obs.obs_id],
                            evidence=[
                                f"{curr.obs_id}: {curr.location} at {curr.timestamp}",
                                f"{next_obs.obs_id}: {next_obs.location} at {next_obs.timestamp}",
                                f"Time gap: {time_diff} seconds (likely too short)",
                            ],
                            recommendation="Review for possible camera/sensor placement or merge error",
                            conservative_action="Flag for review - may be impossible match",
                        )
                        conflicts.append(conflict)

        return conflicts

    def _detect_low_confidence_clusters(
        self,
        clusters: List[EntityCluster],
        cluster_confidences: Dict[str, float],
    ) -> List[DetectedConflict]:
        """
        Detect clusters built on low-confidence edges.

        Args:
            clusters: List of EntityCluster
            cluster_confidences: Dict mapping cluster_id to average edge confidence

        Returns:
            List of DetectedConflict
        """
        conflicts = []

        for cluster in clusters:
            confidence = cluster_confidences.get(cluster.cluster_id, 1.0)

            # Only for multi-observation clusters
            if cluster.size > 1 and confidence < 0.65:
                self._conflict_counter += 1
                conflict = DetectedConflict(
                    conflict_id=f"CFT_{self._conflict_counter:03d}",
                    conflict_type=ConflictType.LOW_CONFIDENCE,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"Cluster {cluster.cluster_id} built on low-confidence edges (avg: {confidence:.3f})",
                    affected_clusters=[cluster.cluster_id],
                    affected_observations=cluster.obs_ids,
                    evidence=[
                        f"Cluster size: {cluster.size}",
                        f"Average edge confidence: {confidence:.3f}",
                        "Threshold for concern: 0.65",
                    ],
                    recommendation="Consider conservative split or manual review of edges",
                    conservative_action="Flag for review - may need re-clustering",
                )
                conflicts.append(conflict)

        return conflicts

    def _detect_suspicious_sizes(
        self, clusters: List[EntityCluster], median_cluster_size: float
    ) -> List[DetectedConflict]:
        """
        Detect unusually large clusters (possible merge errors).

        Args:
            clusters: List of EntityCluster
            median_cluster_size: Median cluster size statistic

        Returns:
            List of DetectedConflict
        """
        conflicts = []

        for cluster in clusters:
            # Flag if cluster is 5+ standard deviations from median
            if cluster.size > 1 and cluster.size > (median_cluster_size * 3):
                self._conflict_counter += 1
                conflict = DetectedConflict(
                    conflict_id=f"CFT_{self._conflict_counter:03d}",
                    conflict_type=ConflictType.SUSPICIOUS_SIZE,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"Cluster {cluster.cluster_id} is unusually large ({cluster.size} observations)",
                    affected_clusters=[cluster.cluster_id],
                    affected_observations=cluster.obs_ids,
                    evidence=[
                        f"Cluster size: {cluster.size}",
                        f"Median cluster size: {median_cluster_size:.1f}",
                        f"Ratio: {cluster.size / max(median_cluster_size, 1):.1f}x median",
                    ],
                    recommendation="Review for possible over-merging",
                    conservative_action="Flag for manual review - possible incorrect merge",
                )
                conflicts.append(conflict)

        return conflicts

    def detect_conflicts(
        self,
        clusters: List[EntityCluster],
        observations: List,
        cluster_confidences: Dict[str, float] = None,
    ) -> Tuple[List[DetectedConflict], ConflictReport]:
        """
        Detect conflicts in clustering result.

        Strategy:
        - Conservative: flag for review rather than auto-fix
        - Multiple detection types for different conflict patterns

        Args:
            clusters: List of EntityCluster
            observations: List of NormalizedObservation
            cluster_confidences: Optional dict of cluster_id -> confidence

        Returns:
            Tuple of (conflicts, report)
        """
        import time
        start_time = time.time()

        self._conflict_counter = 0

        if cluster_confidences is None:
            cluster_confidences = {}

        all_conflicts = []

        # Detect role contradictions
        all_conflicts.extend(self._detect_role_contradictions(clusters, observations))

        # Detect temporal conflicts
        all_conflicts.extend(self._detect_temporal_conflicts(clusters, observations))

        # Detect low-confidence clusters
        all_conflicts.extend(
            self._detect_low_confidence_clusters(clusters, cluster_confidences)
        )

        # Detect suspicious sizes
        sizes = [c.size for c in clusters if c.size > 1]
        median_size = (sum(sizes) / len(sizes)) if sizes else 1.0
        all_conflicts.extend(self._detect_suspicious_sizes(clusters, median_size))

        # Sort by severity
        severity_order = {ConflictSeverity.HIGH: 0, ConflictSeverity.MEDIUM: 1, ConflictSeverity.LOW: 2}
        all_conflicts.sort(key=lambda c: severity_order[c.severity])

        # Compute statistics
        high_sev = sum(1 for c in all_conflicts if c.severity == ConflictSeverity.HIGH)
        med_sev = sum(1 for c in all_conflicts if c.severity == ConflictSeverity.MEDIUM)
        low_sev = sum(1 for c in all_conflicts if c.severity == ConflictSeverity.LOW)

        role_conf = sum(1 for c in all_conflicts if c.conflict_type == ConflictType.ROLE_CONTRADICTION)
        temporal_conf = sum(1 for c in all_conflicts if c.conflict_type == ConflictType.TEMPORAL_CONFLICT)
        low_conf = sum(1 for c in all_conflicts if c.conflict_type == ConflictType.LOW_CONFIDENCE)
        suspicious = sum(1 for c in all_conflicts if c.conflict_type == ConflictType.SUSPICIOUS_SIZE)

        affected_clusters = set()
        affected_obs = set()
        for conflict in all_conflicts:
            affected_clusters.update(conflict.affected_clusters)
            affected_obs.update(conflict.affected_observations)

        elapsed = time.time() - start_time

        report = ConflictReport(
            total_clusters=len(clusters),
            total_observations=len(observations),
            total_conflicts_detected=len(all_conflicts),
            high_severity_conflicts=high_sev,
            medium_severity_conflicts=med_sev,
            low_severity_conflicts=low_sev,
            role_contradiction_count=role_conf,
            temporal_conflict_count=temporal_conf,
            low_confidence_count=low_conf,
            suspicious_size_count=suspicious,
            clusters_with_conflicts=len(affected_clusters),
            observations_affected=len(affected_obs),
            processing_time_sec=elapsed,
        )

        return all_conflicts, report
