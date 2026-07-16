"""
ForenSynth-X+ Generator
Main orchestrator class that implements the full 9-step pipeline:

    1. Select Domain
    2. Select Template
    3. Get template role requirements
    4. Generate FIR  (now template-aware)
    5. Enforce entity counts per template
    6. Create Canonical Entities
    7. Generate Ground Truth Timeline
    8. Expand Events → Clean Observations
    9. Apply Noise → Final Observations
    10. Assemble Case File
    11. Validate case realization
    12. (Optional) Enrich FIR description via LLM
"""

import random
from typing import Optional

from config import GeneratorConfig, NoiseConfig
from domains import DOMAINS, generate_fir
from entities import build_entities, build_entity_mapping, CanonicalEntity
from noise import apply_noise
from observations import expand_events_to_observations
from templates import select_template, get_template_role_requirements
from timeline import generate_timeline, TemplateRealizationError, GroundTruthEvent
from utils import make_case_id, seed_rng, validate_case_schema


def _enforce_template_entity_counts(
    template_requirements: dict[str, int],
    suspect_count: int,
    witness_count: int,
    rng: random.Random,
) -> tuple[int, int]:
    """
    Enforce minimum entity counts based on template requirements.
    
    If the current counts are below template requirements, bump them up.
    Prefers deterministic behavior: if counts are too low, raise an error
    or auto-bump with a clear message.
    
    Args:
        template_requirements: Dict like {"suspect": 3, "witness": 1}
        suspect_count: Current number of suspects
        witness_count: Current number of witnesses
        rng: Random instance (not used in current implementation)
        
    Returns:
        Tuple (final_suspect_count, final_witness_count)
        
    Raises:
        ValueError: If template requirements are fundamentally incompatible
    """
    required_suspects = template_requirements.get("suspect", 0)
    required_witnesses = template_requirements.get("witness", 0)
    
    final_suspects = max(suspect_count, required_suspects)
    final_witnesses = max(witness_count, required_witnesses)
    
    if final_suspects != suspect_count:
        print(
            f"[INFO] Template requires {required_suspects} suspects, "
            f"but {suspect_count} were provided. Auto-bumping to {final_suspects}."
        )
    
    if final_witnesses != witness_count:
        print(
            f"[INFO] Template requires {required_witnesses} witnesses, "
            f"but {witness_count} were provided. Auto-bumping to {final_witnesses}."
        )
    
    return final_suspects, final_witnesses


def _validate_case_realization(
    case: dict,
    template_requirements: dict[str, int],
    entities: list[CanonicalEntity],
    events: list[GroundTruthEvent],
) -> list[str]:
    """
    Validate that the realized case is consistent with template and FIR.
    
    Checks:
    1. All required roles are present in entities
    2. FIR role counts match actual entity counts
    3. Events were actually generated (not empty)
    4. No dangling entity references in events
    
    Args:
        case: The assembled case dict
        template_requirements: Template role requirements
        entities: List of CanonicalEntity objects
        events: List of GroundTruthEvent objects
        
    Returns:
        List of validation error strings (empty = valid)
    """
    errors: list[str] = []
    
    # Check required roles are present
    entity_roles = {e.role for e in entities}
    for required_role, min_count in template_requirements.items():
        if required_role not in entity_roles:
            errors.append(
                f"Template requires '{required_role}' role, but no {required_role} entities exist."
            )
        else:
            role_entities = [e for e in entities if e.role == required_role]
            if len(role_entities) < min_count:
                errors.append(
                    f"Template requires at least {min_count} {required_role}(s), "
                    f"but only {len(role_entities)} exist."
                )
    
    # Check FIR role counts match entity counts
    fir = case.get("fir", {})
    fir_roles = fir.get("roles", {})
    entity_counts: dict[str, int] = {}
    for e in entities:
        entity_counts[e.role] = entity_counts.get(e.role, 0) + 1
    
    for role, fir_count in fir_roles.items():
        actual_count = entity_counts.get(role, 0)
        if fir_count != actual_count:
            errors.append(
                f"FIR claims {fir_count} {role}(s), but {actual_count} exist. "
                f"(FIR should reflect realized case, not random guesses.)"
            )
    
    # Check events are non-empty
    if not events:
        errors.append("No events were generated. Timeline is empty.")
    
    # Check no dangling entity references
    entity_ids = {e.entity_id for e in entities}
    for event in events:
        if event.entity_id not in entity_ids:
            errors.append(
                f"Event {event.event_id} references unknown entity '{event.entity_id}'"
            )
    
    return errors


class ForenSynthGenerator:
    """
    Production-grade synthetic forensic case generator.

    Usage (offline, no API):
        cfg = GeneratorConfig(seed=42)
        gen = ForenSynthGenerator(cfg)
        case = gen.generate_case()

    Usage (with LLM FIR enrichment):
        cfg = GeneratorConfig(
            seed=42,
            enrich_fir=True,
            anthropic_api_key="sk-ant-...",
        )
        gen = ForenSynthGenerator(cfg)
        case = gen.generate_case()

    Args:
        config: GeneratorConfig controlling all generation parameters.
        case_index: Starting numeric index for case IDs (default: 1).
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        case_index: int = 1,
    ) -> None:
        self.config = config or GeneratorConfig()
        self._next_index: int = case_index
        self._rng: random.Random = seed_rng(self.config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_case(
        self,
        domain: Optional[str] = None,
        suspect_count: Optional[int] = None,
        witness_count: Optional[int] = None,
    ) -> dict:
        """
        Execute the full pipeline and return the final case dict.

        Args:
            domain: Force a specific domain (random if None).
            suspect_count: Override number of suspects (1–3 if None).
            witness_count: Override number of witnesses (1–2 if None).

        Returns:
            Case file dict conforming to FINAL CASE FILE SCHEMA.

        Raises:
            ValueError: If the generated case fails validation.
            TemplateRealizationError: If required template slots cannot be realized.
        """
        # ── Step 1: Select Domain ──────────────────────────────────────
        selected_domain: str = domain or self._rng.choice(DOMAINS)

        # ── Step 2: Select Template ────────────────────────────────────
        template = select_template(selected_domain, self._rng)

        # ── Step 3: Get Template Role Requirements ─────────────────────
        template_requirements = get_template_role_requirements(template)

        # ── Step 4: Determine entity counts (enforce template minimum) ─
        n_suspects = suspect_count if suspect_count is not None else self._rng.randint(1, 3)
        n_witnesses = witness_count if witness_count is not None else self._rng.randint(1, 2)
        
        # Enforce template requirements: bump counts if necessary
        n_suspects, n_witnesses = _enforce_template_entity_counts(
            template_requirements=template_requirements,
            suspect_count=n_suspects,
            witness_count=n_witnesses,
            rng=self._rng,
        )

        # ── Step 5: Generate FIR (template-aware) ──────────────────────
        # FIR description pool is specific to the chosen template.
        # FIR role counts now match the enforced entity counts.
        fir = generate_fir(
            domain=selected_domain,
            template_name=template.name,
            suspect_count=n_suspects,
            witness_count=n_witnesses,
            rng=self._rng,
        )

        # ── Step 6: Create Canonical Entities ─────────────────────────
        entities = build_entities(n_suspects, n_witnesses, self._rng)
        entity_mapping = build_entity_mapping(entities)

        # ── Step 7: Generate Ground Truth Timeline ─────────────────────
        # This now enforces required slots and raises on missing entities.
        try:
            events = generate_timeline(
                template=template,
                entities=entities,
                fir=fir,
                base_datetime_str=self.config.base_datetime,
                rng=self._rng,
            )
        except TemplateRealizationError as e:
            raise ValueError(
                f"Failed to realize template '{template.name}' with {n_suspects} suspects "
                f"and {n_witnesses} witnesses:\n{str(e)}"
            )

        # ── Step 8: Expand Events → Clean Observations ─────────────────
        clean_observations = expand_events_to_observations(
            events=events,
            entities=entities,
            template=template,
            rng=self._rng,
            domain=selected_domain,
        )

        # ── Step 9: Apply Noise ────────────────────────────────────────
        event_action_map = {e.event_id: e.action for e in events}
        time_window_sec = fir["time_window"][1] - fir["time_window"][0]
        final_observations = apply_noise(
            clean_observations=clean_observations,
            cfg=self.config.noise,
            rng=self._rng,
            event_action_map=event_action_map,
            time_window_sec=time_window_sec,
        )

        # ── Step 10: Assemble Case File ─────────────────────────────────
        case_id = make_case_id(selected_domain, self._next_index)
        self._next_index += 1

        case = {
            "case_id": case_id,
            "domain": selected_domain,
            "template": template.name,
            "fir": fir,
            "observations": final_observations,
            "ground_truth": {
                "entities": [e.to_dict() for e in entities],
                "events": [e.to_dict() for e in events],
                "entity_mapping": entity_mapping,
            },
        }

        # ── Step 11: Validate case realization ────────────────────────
        validation_errors = _validate_case_realization(
            case=case,
            template_requirements=template_requirements,
            entities=entities,
            events=events,
        )
        if validation_errors:
            raise ValueError(
                f"Case '{case_id}' failed realization validation:\n" +
                "\n".join(f"  - {e}" for e in validation_errors)
            )

        # ── Step 12 (Optional): Single-call LLM enrichment ─────────────
        if self.config.enrich and self.config.cohere_api_key:
            from enrichment import enrich_case
            case = enrich_case(
                case=case,
                api_key=self.config.cohere_api_key,
            )

        # ── Final Schema Validation ────────────────────────────────────
        errors = validate_case_schema(case)
        if errors:
            raise ValueError("Case schema validation failed:\n" + "\n".join(errors))

        return case

    def generate_batch(
        self,
        n: int,
        domain: Optional[str] = None,
    ) -> list[dict]:
        """
        Generate a batch of n cases.

        Each case increments the internal case index and uses successive RNG
        state — fully reproducible given the same initial seed and case_index.

        Args:
            n: Number of cases to generate.
            domain: If set, all cases use this domain.

        Returns:
            List of case dicts.
        """
        return [self.generate_case(domain=domain) for _ in range(n)]
