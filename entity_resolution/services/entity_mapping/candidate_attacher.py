"""Candidate attachment service for entity resolution.

Stage 9: Links unresolved observations to clusters via candidate edges.

Strategy:
- Candidate edges (0.60-0.80) represent moderate-confidence pairs
- Instead of merging clusters, create explicit attachments
- Allow manual review without forcing merges
- Provides traceability for borderline decisions

Principle:
- Singletons (isolated observations) can be attached to clusters
- DOES NOT merge clusters (keeps decision hierarchy clear)
- Each attachment includes confidence, reasoning, and conflict flags

Output: List of attachments + statistics
"""

from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum

from .clusterer import EntityCluster
from .edge_classifier import ClassifiedEdge, EdgeClassification


# ============================================================================
# Data Structures
# ============================================================================


class AttachmentReason(str, Enum):
    """Reason for attachment decision."""

    CANDIDATE_EDGE = "candidate_edge"  # Moderate confidence pairing
    TEMPORAL_PROXIMITY = "temporal_proximity"  # Close in time
    LOCATION_MATCH = "location_match"  # Same location
    ROLE_COMPATIBLE = "role_compatible"  # Compatible roles


@dataclass
class ConflictFlag:
    """Potential conflict associated with attachment."""

    conflict_type: str  # e.g., "role_mismatch", "temporal_conflict"
    severity: str  # "low", "medium", "high"
    description: str
    recommendation: str


@dataclass
class CandidateAttachment:
    """Attachment of observation to cluster via candidate edge."""

    obs_id: str  # Observation ID
    cluster_id: str  # Target cluster ID
    similarity_score: float  # [0.0, 1.0]
    confidence: str  # "high_candidate", "medium_candidate", "low_candidate"
    
    reasons: List[AttachmentReason] = field(default_factory=list)
    conflict_flags: List[ConflictFlag] = field(default_factory=list)
    
    # Context
    target_cluster_size: int = 0  # Size of cluster being attached to
    potential_entity_roles: List[str] = field(default_factory=list)  # Roles in cluster
    observation_role: str = ""  # Role of observation

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "obs_id": self.obs_id,
            "cluster_id": self.cluster_id,
            "similarity_score": round(self.similarity_score, 3),
            "confidence": self.confidence,
            "reasons": [r.value for r in self.reasons],
            "conflicts": [
                {
                    "type": cf.conflict_type,
                    "severity": cf.severity,
                    "description": cf.description,
                    "recommendation": cf.recommendation,
                }
                for cf in self.conflict_flags
            ],
            "target_cluster_size": self.target_cluster_size,
            "target_cluster_roles": self.potential_entity_roles,
            "observation_role": self.observation_role,
        }


@dataclass
class AttachmentReport:
    """Report from candidate attachment stage."""

    total_candidate_edges: int = 0
    total_attachments_created: int = 0  # Final attachment count
    
    # Attachment distribution
    high_confidence_attachments: int = 0  # 0.75-0.80
    medium_confidence_attachments: int = 0  # 0.65-0.75
    low_confidence_attachments: int = 0  # 0.60-0.65
    
    # Conflict tracking
    attachments_with_conflicts: int = 0
    total_conflicts_detected: int = 0
    high_severity_conflicts: int = 0
    medium_severity_conflicts: int = 0
    low_severity_conflicts: int = 0
    
    processing_time_sec: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "total_candidate_edges": self.total_candidate_edges,
            "total_attachments": self.total_attachments_created,
            "high_confidence": self.high_confidence_attachments,
            "medium_confidence": self.medium_confidence_attachments,
            "low_confidence": self.low_confidence_attachments,
            "attachments_with_conflicts": self.attachments_with_conflicts,
            "total_conflicts": self.total_conflicts_detected,
            "high_severity": self.high_severity_conflicts,
            "medium_severity": self.medium_severity_conflicts,
            "low_severity": self.low_severity_conflicts,
            "processing_time_sec": round(self.processing_time_sec, 3),
        }


# ============================================================================
# Candidate Attachment Implementation
# ============================================================================


class CandidateAttacher:
    """Attaches candidate observations to clusters without merging."""

    def __init__(self):
        """Initialize attacher."""
        pass

    def _get_cluster_roles(self, cluster: EntityCluster, observations: List) -> Set[str]:
        """
        Extract roles from all observations in a cluster.

        Args:
            cluster: EntityCluster
            observations: List of NormalizedObservation

        Returns:
            Set of role strings in cluster
        """
        roles = set()
        obs_dict = {obs.obs_id: obs for obs in observations}
        
        for obs_id in cluster.obs_ids:
            obs = obs_dict.get(obs_id)
            if obs and hasattr(obs, 'role'):
                roles.add(obs.role)
        
        return roles

    def _detect_attachment_conflicts(
        self,
        obs_id: str,
        obs_role: str,
        cluster: EntityCluster,
        cluster_roles: Set[str],
        similarity_score: float,
    ) -> List[ConflictFlag]:
        """
        Detect potential conflicts for an attachment.

        Args:
            obs_id: Observation ID to attach
            obs_role: Role of observation
            cluster: Target cluster
            cluster_roles: Set of roles in cluster
            similarity_score: Similarity score for attachment

        Returns:
            List of ConflictFlag
        """
        conflicts = []

        # Conflict 1: Role mismatch
        if obs_role and cluster_roles:
            if obs_role not in cluster_roles:
                # Different role - could be conflict
                if obs_role == "system" or "system" in cluster_roles:
                    # System observations are generic
                    conflicts.append(
                        ConflictFlag(
                            conflict_type="role_type_mismatch",
                            severity="low",
                            description=f"Observation has role '{obs_role}', cluster has roles {cluster_roles}",
                            recommendation="Review for manual entry system observations",
                        )
                    )
                else:
                    # Non-system role mismatch is more concerning
                    conflicts.append(
                        ConflictFlag(
                            conflict_type="role_contradiction",
                            severity="medium",
                            description=f"Observation role '{obs_role}' differs from cluster roles {cluster_roles}",
                            recommendation="Manual review required - may indicate different entities",
                        )
                    )

        # Conflict 2: Low confidence attachment
        if similarity_score < 0.65:
            conflicts.append(
                ConflictFlag(
                    conflict_type="low_confidence",
                    severity="low",
                    description=f"Similarity score {similarity_score:.3f} is at lower end of candidate range",
                    recommendation="Consider manual review for accuracy",
                )
            )

        # Conflict 3: Borderline candidate (very close to confirmed threshold)
        if similarity_score >= 0.77:
            # This is actually good - very close to confirmed
            pass  # No conflict
        elif similarity_score >= 0.72:
            # Close to confirmed, relatively high confidence
            pass  # No conflict
        else:
            # Further from confirmed - increased uncertainty
            if similarity_score < 0.65:
                conflicts.append(
                    ConflictFlag(
                        conflict_type="distance_from_thresholds",
                        severity="low",
                        description=f"Score {similarity_score:.3f} distant from both thresholds (0.60, 0.80)",
                        recommendation="Review both attachment recommendation and cluster merge option",
                    )
                )

        return conflicts

    def attach_candidates(
        self,
        candidate_edges: List[ClassifiedEdge],
        clusters: List[EntityCluster],
        observations: List,
    ) -> Tuple[List[CandidateAttachment], AttachmentReport]:
        """
        Create attachments for candidate edges.

        Strategy:
        - Non-merged observation (singleton) can be attached to cluster
        - Attachment provides link for manual review
        - Does NOT merge clusters (keeps decision hierarchy clear)

        Args:
            candidate_edges: List of ClassifiedEdge with CANDIDATE classification
            clusters: List of EntityCluster from Stage 8
            observations: List of NormalizedObservation

        Returns:
            Tuple of (attachments, report)
            - attachments: List of CandidateAttachment
            - report: AttachmentReport with statistics
        """
        import time
        start_time = time.time()

        # Build cluster map
        cluster_map = {obs_id: cluster for cluster in clusters for obs_id in cluster.obs_ids}
        obs_dict = {obs.obs_id: obs for obs in observations}

        attachments: List[CandidateAttachment] = []
        conflict_count = 0
        high_sev_conflicts = 0
        med_sev_conflicts = 0
        low_sev_conflicts = 0

        # Filter to only CANDIDATE edges
        candidate_edges_only = [
            e for e in candidate_edges if e.classification == EdgeClassification.CANDIDATE
        ]

        for edge in candidate_edges_only:
            obs_1 = obs_dict.get(edge.obs_id_1)
            obs_2 = obs_dict.get(edge.obs_id_2)

            if not obs_1 or not obs_2:
                continue

            cluster_1 = cluster_map.get(edge.obs_id_1)
            cluster_2 = cluster_map.get(edge.obs_id_2)

            # Case 1: obs_1 is singleton, obs_2 is in a cluster
            if cluster_1 and cluster_1.size == 1 and cluster_2 and cluster_2.size > 1:
                cluster_roles = self._get_cluster_roles(cluster_2, observations)
                obs_role = obs_1.role if hasattr(obs_1, 'role') else ""

                conflicts = self._detect_attachment_conflicts(
                    edge.obs_id_1, obs_role, cluster_2, cluster_roles, edge.similarity_score
                )

                confidence = self._classify_confidence(edge.similarity_score)
                reasons = [AttachmentReason.CANDIDATE_EDGE]

                attachment = CandidateAttachment(
                    obs_id=edge.obs_id_1,
                    cluster_id=cluster_2.cluster_id,
                    similarity_score=edge.similarity_score,
                    confidence=confidence,
                    reasons=reasons,
                    conflict_flags=conflicts,
                    target_cluster_size=cluster_2.size,
                    potential_entity_roles=sorted(list(cluster_roles)),
                    observation_role=obs_role,
                )

                attachments.append(attachment)
                if conflicts:
                    conflict_count += 1
                    for cf in conflicts:
                        if cf.severity == "high":
                            high_sev_conflicts += 1
                        elif cf.severity == "medium":
                            med_sev_conflicts += 1
                        else:
                            low_sev_conflicts += 1

            # Case 2: obs_2 is singleton, obs_1 is in a cluster
            elif cluster_2 and cluster_2.size == 1 and cluster_1 and cluster_1.size > 1:
                cluster_roles = self._get_cluster_roles(cluster_1, observations)
                obs_role = obs_2.role if hasattr(obs_2, 'role') else ""

                conflicts = self._detect_attachment_conflicts(
                    edge.obs_id_2, obs_role, cluster_1, cluster_roles, edge.similarity_score
                )

                confidence = self._classify_confidence(edge.similarity_score)
                reasons = [AttachmentReason.CANDIDATE_EDGE]

                attachment = CandidateAttachment(
                    obs_id=edge.obs_id_2,
                    cluster_id=cluster_1.cluster_id,
                    similarity_score=edge.similarity_score,
                    confidence=confidence,
                    reasons=reasons,
                    conflict_flags=conflicts,
                    target_cluster_size=cluster_1.size,
                    potential_entity_roles=sorted(list(cluster_roles)),
                    observation_role=obs_role,
                )

                attachments.append(attachment)
                if conflicts:
                    conflict_count += 1
                    for cf in conflicts:
                        if cf.severity == "high":
                            high_sev_conflicts += 1
                        elif cf.severity == "medium":
                            med_sev_conflicts += 1
                        else:
                            low_sev_conflicts += 1

        # Sort by similarity score (descending)
        attachments.sort(key=lambda a: a.similarity_score, reverse=True)

        # Count by confidence
        high_conf = sum(1 for a in attachments if a.confidence == "high_candidate")
        med_conf = sum(1 for a in attachments if a.confidence == "medium_candidate")
        low_conf = sum(1 for a in attachments if a.confidence == "low_candidate")

        elapsed = time.time() - start_time

        report = AttachmentReport(
            total_candidate_edges=len(candidate_edges_only),
            total_attachments_created=len(attachments),
            high_confidence_attachments=high_conf,
            medium_confidence_attachments=med_conf,
            low_confidence_attachments=low_conf,
            attachments_with_conflicts=conflict_count,
            total_conflicts_detected=high_sev_conflicts + med_sev_conflicts + low_sev_conflicts,
            high_severity_conflicts=high_sev_conflicts,
            medium_severity_conflicts=med_sev_conflicts,
            low_severity_conflicts=low_sev_conflicts,
            processing_time_sec=elapsed,
        )

        return attachments, report

    @staticmethod
    def _classify_confidence(similarity_score: float) -> str:
        """
        Classify attachment confidence based on similarity score.

        Args:
            similarity_score: [0.6, 0.8] (candidate range)

        Returns:
            Confidence level string
        """
        if similarity_score >= 0.75:
            return "high_candidate"
        elif similarity_score >= 0.65:
            return "medium_candidate"
        else:
            return "low_candidate"
