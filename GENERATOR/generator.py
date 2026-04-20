"""
ForenSynth-X+ Generator
Main orchestrator class that implements the full 9-step pipeline:

    1. Select Domain
    2. Select Template
    3. Generate FIR  (now template-aware)
    4. Define Roles
    5. Create Canonical Entities
    6. Generate Ground Truth Timeline
    7. Expand Events → Clean Observations
    8. Apply Noise → Final Observations
    9. Assemble Case File
   10. (Optional) Enrich FIR description via LLM
"""

import random
from typing import Optional

from config import GeneratorConfig, NoiseConfig
from domains import DOMAINS, generate_fir
from entities import build_entities, build_entity_mapping
from noise import apply_noise
from observations import expand_events_to_observations
from templates import select_template
from timeline import generate_timeline
from utils import make_case_id, seed_rng, validate_case_schema


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
            ValueError: If the generated case fails schema validation.
        """
        # ── Step 1: Select Domain ──────────────────────────────────────
        selected_domain: str = domain or self._rng.choice(DOMAINS)

        # ── Step 2: Select Template ────────────────────────────────────
        template = select_template(selected_domain, self._rng)

        # ── Steps 3 & 4: Generate FIR + Define Roles ──────────────────
        # FIR is now template-aware: description pool is specific to the
        # chosen template, not just the domain.
        n_suspects = suspect_count if suspect_count is not None else self._rng.randint(1, 3)
        n_witnesses = witness_count if witness_count is not None else self._rng.randint(1, 2)
        fir = generate_fir(
            domain=selected_domain,
            template_name=template.name,       # ← template-aware
            suspect_count=n_suspects,
            witness_count=n_witnesses,
            rng=self._rng,
        )

        # ── Step 5: Create Canonical Entities ─────────────────────────
        entities = build_entities(n_suspects, n_witnesses, self._rng)
        entity_mapping = build_entity_mapping(entities)

        # ── Step 6: Generate Ground Truth Timeline ─────────────────────
        events = generate_timeline(
            template=template,
            entities=entities,
            fir=fir,
            base_datetime_str=self.config.base_datetime,
            rng=self._rng,
        )

        # ── Step 7: Expand Events → Clean Observations ─────────────────
        clean_observations = expand_events_to_observations(
            events=events,
            entities=entities,
            template=template,
            rng=self._rng,
            domain=selected_domain,
        )

        # ── Step 8: Apply Noise ────────────────────────────────────────
        # Pass event_id → action map so contradiction lookup is exact,
        # not a fragile content-keyword heuristic.
        event_action_map = {e.event_id: e.action for e in events}
        final_observations = apply_noise(
            clean_observations=clean_observations,
            cfg=self.config.noise,
            rng=self._rng,
            event_action_map=event_action_map,
        )

        # ── Step 9: Assemble Case File ─────────────────────────────────
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

        # ── Step 10 (Optional): Single-call Cohere enrichment ─────────
        # One API call rewrites: fir.description, fir.location,
        # observations[].content, observations[].source.
        # Ground truth, timestamps, noise_tags, aliases: never touched.
        if self.config.enrich and self.config.cohere_api_key:
            from enrichment import enrich_case
            case = enrich_case(
                case=case,
                api_key=self.config.cohere_api_key,
            )

        # ── Validate ───────────────────────────────────────────────────
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
