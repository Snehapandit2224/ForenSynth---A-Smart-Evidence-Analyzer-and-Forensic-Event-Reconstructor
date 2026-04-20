"""Entity labeler for the resolution pipeline.

Stage 11: Assign entity IDs and aggregate canonical entity information.

Strategy:
- Assign sequential entity IDs (entity_1, entity_2, ...)
- Aggregate all observations linked to each entity
- Compute confidence scores based on confirmed/candidate links
- Build comprehensive entity profiles

Output: List of CanonicalEntity with full metadata
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .clusterer import EntityCluster
from ...schemas.observation import NormalizedObservation, Modality


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class CanonicalEntity:
    """A canonical entity with aggregated information."""

    entity_id: str  # e.g., "entity_1", "entity_2"
    
    # Identity
    aliases: Set[str] = field(default_factory=set)  # All entity aliases seen
    primary_alias: str = ""  # Most common or first alias
    
    # Observations
    confirmed_mentions: List[str] = field(default_factory=list)  # Obs IDs with confirmed edges
    candidate_mentions: List[str] = field(default_factory=list)  # Obs IDs with candidate edges
    total_mention_count: int = 0
    
    # Confidence
    confidence_score: float = 0.0  # [0.0, 1.0] - overall entity confidence
    confirmed_edge_count: int = 0  # Number of confirmed connections
    candidate_edge_count: int = 0  # Number of candidate connections
    
    # Attributes
    modalities: Set[str] = field(default_factory=set)  # All modalities observed
    locations: Set[str] = field(default_factory=set)  # All locations observed
    roles: Set[str] = field(default_factory=set)  # All roles observed
    sources: Set[str] = field(default_factory=set)  # All data sources
    
    # Temporal
    earliest_timestamp: Optional[datetime] = None
    latest_timestamp: Optional[datetime] = None
    time_span_sec: int = 0  # Duration from first to last mention
    
    # Content
    content_snippets: List[str] = field(default_factory=list)  # Sample descriptions
    noise_tags: Set[str] = field(default_factory=set)  # All noise/quality issues

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "entity_id": self.entity_id,
            "aliases": sorted(list(self.aliases)),
            "primary_alias": self.primary_alias,
            "total_mentions": self.total_mention_count,
            "confirmed_mentions": self.confirmed_mentions,
            "candidate_mentions": self.candidate_mentions,
            "confidence_score": round(self.confidence_score, 3),
            "confirmed_edges": self.confirmed_edge_count,
            "candidate_edges": self.candidate_edge_count,
            "modalities": sorted(list(self.modalities)),
            "locations": sorted(list(self.locations)),
            "roles": sorted(list(self.roles)),
            "sources": sorted(list(self.sources)),
            "earliest_timestamp": self.earliest_timestamp.isoformat() if self.earliest_timestamp else None,
            "latest_timestamp": self.latest_timestamp.isoformat() if self.latest_timestamp else None,
            "time_span_seconds": self.time_span_sec,
        }


@dataclass
class LabelingReport:
    """Report from labeling stage."""

    total_clusters: int = 0
    total_entities_created: int = 0
    
    # Entity size distribution
    singleton_entities: int = 0  # Size 1
    pair_entities: int = 0  # Size 2
    triplet_entities: int = 0  # Size 3
    large_entities: int = 0  # Size >= 4
    
    # Confidence distribution
    high_confidence_entities: int = 0  # >= 0.80
    medium_confidence_entities: int = 0  # 0.60-0.80
    low_confidence_entities: int = 0  # < 0.60
    
    # Edge statistics
    total_confirmed_edges: int = 0
    total_candidate_edges: int = 0
    
    # Modality coverage
    video_entities: int = 0
    audio_entities: int = 0
    text_entities: int = 0
    multimodal_entities: int = 0  # More than one modality
    
    # Coverage
    total_mentions: int = 0
    avg_mentions_per_entity: float = 0.0
    
    processing_time_sec: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "total_clusters": self.total_clusters,
            "total_entities": self.total_entities_created,
            "singletons": self.singleton_entities,
            "pairs": self.pair_entities,
            "triplets": self.triplet_entities,
            "large_clusters": self.large_entities,
            "high_confidence": self.high_confidence_entities,
            "medium_confidence": self.medium_confidence_entities,
            "low_confidence": self.low_confidence_entities,
            "total_confirmed_edges": self.total_confirmed_edges,
            "total_candidate_edges": self.total_candidate_edges,
            "video_entities": self.video_entities,
            "audio_entities": self.audio_entities,
            "text_entities": self.text_entities,
            "multimodal_entities": self.multimodal_entities,
            "total_mentions": self.total_mentions,
            "avg_mentions_per_entity": round(self.avg_mentions_per_entity, 2),
            "processing_time_sec": round(self.processing_time_sec, 3),
        }


# ============================================================================
# Labeler Implementation
# ============================================================================


class Labeler:
    """Assigns entity identifiers and aggregates entity information."""

    def __init__(self):
        """Initialize labeler."""
        pass

    def _compute_confidence_score(
        self,
        cluster_size: int,
        confirmed_edge_count: int,
        candidate_edge_count: int,
        avg_observation_confidence: float,
    ) -> float:
        """
        Compute confidence score for an entity.

        Factors:
        - Cluster size (more mentions = higher confidence)
        - Confirmed edges (high-confidence links)
        - Candidate edges (moderate-confidence links)
        - Observation confidences (input quality)

        Args:
            cluster_size: Number of observations in cluster
            confirmed_edge_count: Number of confirmed edges
            candidate_edge_count: Number of candidate edges
            avg_observation_confidence: Average observation confidence

        Returns:
            Confidence score [0.0, 1.0]
        """
        # Base confidence from input observations
        base_conf = avg_observation_confidence

        # Boost from confirmed edges (strong linking)
        confirmed_boost = min(confirmed_edge_count * 0.05, 0.15)  # Max 0.15 boost

        # Mild boost from candidate edges (weaker linking)
        candidate_boost = min(candidate_edge_count * 0.01, 0.05)  # Max 0.05 boost

        # Size boost (multiple observations = more likely correct)
        size_boost = min((cluster_size - 1) * 0.02, 0.10)  # Max 0.10 boost

        # Combine with cap at 1.0
        combined = base_conf + confirmed_boost + candidate_boost + size_boost
        return min(combined, 1.0)

    def label_entities(
        self,
        clusters: List[EntityCluster],
        observations: List[NormalizedObservation],
        confirmed_edge_count: int = 0,
        candidate_edge_count: int = 0,
    ) -> Tuple[List[CanonicalEntity], LabelingReport]:
        """
        Label clusters as canonical entities.

        Args:
            clusters: List of EntityCluster
            observations: List of NormalizedObservation
            confirmed_edge_count: Total confirmed edges in graph
            candidate_edge_count: Total candidate edges (for statistics)

        Returns:
            Tuple of (canonical_entities, report)
        """
        import time

        start_time = time.time()

        # Build obs_id -> observation map
        obs_dict = {obs.obs_id: obs for obs in observations}

        # Create entities from clusters
        canonical_entities = []
        entity_counter = 1

        for cluster in clusters:
            entity_id = f"entity_{entity_counter}"
            entity_counter += 1

            # Collect all observations in cluster
            cluster_obs = []
            for obs_id in cluster.obs_ids:
                obs = obs_dict.get(obs_id)
                if obs:
                    cluster_obs.append(obs)

            if not cluster_obs:
                continue

            # Extract identity information
            aliases = set()
            roles = set()
            modalities = set()
            locations = set()
            sources = set()
            content_snippets = []
            noise_tags = set()
            timestamps = []

            for obs in cluster_obs:
                aliases.add(obs.entity)
                if hasattr(obs, "role"):
                    roles.add(obs.role)

                # Modality
                modality = obs.modality
                if isinstance(modality, str):
                    modalities.add(modality)
                elif isinstance(modality, Modality):
                    modalities.add(modality.value)
                else:
                    modalities.add(str(modality))

                locations.add(obs.location)
                sources.add(obs.source)
                content_snippets.append(obs.content[:50])  # First 50 chars
                noise_tags.update(obs.noise_tags)
                timestamps.append(obs.timestamp_dt)

            # Compute temporal span
            valid_timestamps = [ts for ts in timestamps if ts is not None]
            earliest_ts = None
            latest_ts = None
            time_span = 0

            if valid_timestamps:
                earliest_ts = min(valid_timestamps)
                latest_ts = max(valid_timestamps)
                time_span = int((latest_ts - earliest_ts).total_seconds())

            # Compute confidence
            avg_obs_conf = sum(obs.confidence for obs in cluster_obs) / len(cluster_obs)
            edge_count = 1 if len(cluster_obs) > 1 else 0  # At least 1 edge if >1 obs
            confidence = self._compute_confidence_score(
                cluster_size=len(cluster_obs),
                confirmed_edge_count=edge_count,
                candidate_edge_count=0,
                avg_observation_confidence=avg_obs_conf,
            )

            # Create entity
            entity = CanonicalEntity(
                entity_id=entity_id,
                aliases=aliases,
                primary_alias=sorted(list(aliases))[0],
                confirmed_mentions=cluster.obs_ids if len(cluster_obs) > 1 else [],
                candidate_mentions=[],
                total_mention_count=len(cluster_obs),
                confidence_score=confidence,
                confirmed_edge_count=edge_count if len(cluster_obs) > 1 else 0,
                candidate_edge_count=0,
                modalities=modalities,
                locations=locations,
                roles=roles,
                sources=sources,
                earliest_timestamp=earliest_ts,
                latest_timestamp=latest_ts,
                time_span_sec=time_span,
                content_snippets=content_snippets[:3],  # First 3 snippets
                noise_tags=noise_tags,
            )

            canonical_entities.append(entity)

        # Compute statistics
        high_conf = sum(1 for e in canonical_entities if e.confidence_score >= 0.80)
        med_conf = sum(1 for e in canonical_entities if 0.60 <= e.confidence_score < 0.80)
        low_conf = sum(1 for e in canonical_entities if e.confidence_score < 0.60)

        singleton_count = sum(1 for e in canonical_entities if e.total_mention_count == 1)
        pair_count = sum(1 for e in canonical_entities if e.total_mention_count == 2)
        triplet_count = sum(1 for e in canonical_entities if e.total_mention_count == 3)
        large_count = sum(1 for e in canonical_entities if e.total_mention_count >= 4)

        video_entities = sum(1 for e in canonical_entities if "video" in e.modalities)
        audio_entities = sum(1 for e in canonical_entities if "audio" in e.modalities)
        text_entities = sum(1 for e in canonical_entities if "text" in e.modalities)
        multimodal = sum(1 for e in canonical_entities if len(e.modalities) > 1)

        total_mentions = sum(e.total_mention_count for e in canonical_entities)
        avg_mentions = (
            total_mentions / len(canonical_entities) if canonical_entities else 0.0
        )

        elapsed = time.time() - start_time

        report = LabelingReport(
            total_clusters=len(clusters),
            total_entities_created=len(canonical_entities),
            singleton_entities=singleton_count,
            pair_entities=pair_count,
            triplet_entities=triplet_count,
            large_entities=large_count,
            high_confidence_entities=high_conf,
            medium_confidence_entities=med_conf,
            low_confidence_entities=low_conf,
            total_confirmed_edges=confirmed_edge_count,
            total_candidate_edges=candidate_edge_count,
            video_entities=video_entities,
            audio_entities=audio_entities,
            text_entities=text_entities,
            multimodal_entities=multimodal,
            total_mentions=total_mentions,
            avg_mentions_per_entity=avg_mentions,
            processing_time_sec=elapsed,
        )

        return canonical_entities, report
