import sys, json
sys.path.insert(0, 'agents')
sys.path.insert(0, 'memory')

from memory_store import ForenSynthMemory
mem = ForenSynthMemory()

versions = mem.get_latest_versions('CASE_ATM_002')
print('Versions in DB:', versions)

payload = mem.load_for_showrunner('CASE_ATM_002', tl_version='V1', crit_version='C1')
critique = payload['critique']
print('requires_revision:', critique.get('requires_revision'))
print('revision_target:', critique.get('revision_target'))
print('recommended_action:', critique.get('recommended_action'))
print()
print('Gaps:')
for g in critique.get('gaps', []):
    ht = g.get('fix_hint','')[:60]
    print(f"  {g['gap_type']}  fix_hint: {ht}")

from showrunner_agent import run_showrunner
decision = run_showrunner(payload)
print()
print('Showrunner action:', decision['action'])
print('Reasoning:', decision['reasoning'][:150])
