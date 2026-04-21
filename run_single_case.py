import json
from pathlib import Path

from entity_resolution.services.entity_mapping.resolver import Resolver


def run_one_case(obs_json_file):

    obs_file = Path(obs_json_file)

    if not obs_file.exists():
        raise FileNotFoundError(
            f"{obs_json_file} not found"
        )

    with open(obs_file,"r",encoding="utf-8") as f:
        case_data = json.load(f)

    resolver = Resolver()

    result = resolver.resolve_case_dict(case_data)

    print("\n=== ENTITY RESOLUTION RESULT ===")
    print("Case:", result.case_id)
    print("Status:", result.status)
    print("Entities found:", result.entity_count)

    print("\nCanonical Entities:")
    for e in result.canonical_entities:
        if hasattr(e,"to_dict"):
            print(e.to_dict())
        else:
            print(e)

    output_file = obs_file.parent / "entity_resolution_output1.json"

    with open(output_file,"w",encoding="utf-8") as f:
        json.dump(result.to_dict(),f,indent=2)

    print(f"\nSaved results to {output_file}")


if __name__=="__main__":

    CASE_TO_RUN = r"C:\Users\suche\Desktop\Capstone\ForenSynth---A-Smart-Evidence-Analyzer-and-Forensic-Event-Reconstructor\GENERATOR_FIXED\cases_office\CASE_OFF_001_obs_only.json"

    run_one_case(CASE_TO_RUN)