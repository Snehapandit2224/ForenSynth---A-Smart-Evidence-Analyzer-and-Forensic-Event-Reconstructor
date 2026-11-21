"""
The agent loop: the model looks at the clip, calls tools until it is confident, then gives a
final answer. The final answer must pass the field-rule checks (validate); if it doesn't, the
problems are sent back and the model fixes them (up to GATE_ROUNDS times).
"""
import json

from .backends import parse_json
from .tools import validate

GATE_ROUNDS = 2


def run_loop(backend, clip, fps, system, user, tools, ctx, max_steps=8, log=print):
    turns = [{"role": "user", "parts": [{"video": str(clip), "fps": fps}, {"text": user}]}]
    trace, gates = [], 0

    def ask():
        text = backend.chat(system, turns)
        turns.append({"role": "model", "parts": [{"text": text}]})
        try:
            return parse_json(text)
        except ValueError:
            return {"_unparseable": text}

    for step in range(max_steps):
        reply = ask()
        thought = reply.get("thought", "")
        if "final" in reply and isinstance(reply["final"], dict):
            final = reply["final"]
            issues = validate(final, tools.vocab, tools.registry, ctx, tools.verified_clock)
            trace.append({"step": step, "thought": thought, "final": True, "issues": issues})
            if issues and gates < GATE_ROUNDS and step < max_steps - 1:
                gates += 1
                log(f"    step {step}: final answer rejected ({len(issues)} problems), sent back")
                turns.append({"role": "user", "parts": [{"text":
                    "Your final answer failed these checks. Fix them (use tools if needed) and answer again:\n- "
                    + "\n- ".join(issues)}]})
                continue
            log(f"    step {step}: final answer" + (f" ({len(issues)} unresolved problems)" if issues else ""))
            return final, trace, issues
        if "tool" in reply:
            name, args = reply["tool"], reply.get("args") or {}
            result = tools.call(name, args)
            short = (result.get("text") or "").splitlines()[0][:110]
            log(f"    step {step}: {name}({json.dumps(args)[:80]}) -> {short}")
            trace.append({"step": step, "thought": thought, "tool": name, "args": args, "result": result.get("text")})
            parts = [{"text": f"TOOL RESULT ({name}):\n{result.get('text', '')}"}]
            parts += [{"image": img} for img in result.get("images", [])]
            if result.get("video"):
                parts.append({"video": result["video"], "fps": result.get("fps", fps)})
            turns.append({"role": "user", "parts": parts})
            continue
        trace.append({"step": step, "invalid_reply": str(reply)[:300]})
        turns.append({"role": "user", "parts": [{"text":
            'Reply with ONE JSON object: {"thought": ..., "tool": ..., "args": {...}} or {"thought": ..., "final": {...}}.'}]})

    turns.append({"role": "user", "parts": [{"text": "Step limit reached. Give your final answer now, no more tools."}]})
    reply = ask()
    final = reply.get("final") if isinstance(reply.get("final"), dict) else {}
    issues = validate(final, tools.vocab, tools.registry, ctx, tools.verified_clock)
    trace.append({"step": max_steps, "final": True, "forced": True, "issues": issues})
    log(f"    step limit: forced final answer ({len(issues)} problems)")
    return final, trace, issues
