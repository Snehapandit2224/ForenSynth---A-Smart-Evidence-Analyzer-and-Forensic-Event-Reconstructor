import sys, json
sys.path.insert(0, "memory")
from memory_store import ForenSynthMemory
mem = ForenSynthMemory()
row = mem._query_one(
    "SELECT full_json FROM critique_runs WHERE case_id=? AND critique_version=?",
    ("CASE_ATM_001", "C1")
)
critique = json.loads(row["full_json"])
print("llm_used:", critique.get("llm_used"))
print("checks_run:", critique.get("checks_run"))
for i in critique.get("issues", []):
    print(f"  [{i.get('severity')}] {i.get('check')}  detail={i.get('detail','')[:80]}")