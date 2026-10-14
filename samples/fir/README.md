# FIR test samples

All people, places-of-business, phone numbers and case numbers here are **fictional**,
hand-written to test `agents/field_agent/fir_agent.py` and `agents/field_agent/text_agent.py`.

| File | What it tests |
|---|---|
| `FIR_001_chain_snatching.txt` | Form-style FIR (numbered BNSS sections): anchors, unknown accused with descriptions, property, delay reason |
| `FIR_001_witness_statement.txt` | Witness statement, same incident. **Deliberate conflict:** gives the time as 8:30-8:40 pm; the FIR says ~8:45 pm |
| `FIR_001_chat.json` | Chat export (JSON message list): planning, admission, a fence, an instruction to delete messages. Named suspects the FIR doesn't know about |
| `FIR_002_house_burglary.txt` | Free-narrative complaint letter (no form): relative dates ("yesterday"), hearsay from two witnesses, unknown accused |

Run case 1 end to end:

```bash
python agents/field_agent/orchestrator.py --case-id CASE_FIR_001 \
    --fir samples/fir/FIR_001_chain_snatching.txt \
    --text-inputs samples/fir/FIR_001_witness_statement.txt samples/fir/FIR_001_chat.json
```

Run case 2, FIR only:

```bash
python agents/field_agent/fir_agent.py --case-id CASE_FIR_002 --input samples/fir/FIR_002_house_burglary.txt
```
