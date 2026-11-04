"""
The agent loop, capped at MAX_API_CALLS model requests per clip (3).

  call 1  the model watches the clip and replies with its final answer, or with a BATCH of tool
          calls ({"tools": [...]}) that are all run locally, in order, before the next call.
  call 2  the model gets the tool results and must give its final answer; no more tools.
  call 3  only if the last final answer failed the field-rule checks: the problems are sent back
          and the model fixes them. A clip whose answer passes never uses it.

Tools are only run for call 1. A final answer that passes validate ends the clip at once (after
one request if the model answered straight away). Problems still present after the last call are
returned as unresolved issues. Rate-limit retries inside the backend repeat the same request and
are not extra calls.
"""
import json

from .backends import parse_json
from .tools import validate

MAX_API_CALLS = 3
MAX_TOOLS_PER_BATCH = 8


def run_loop(backend, clip, fps, system, user, tools, ctx, max_calls=MAX_API_CALLS, log=print):
    max_calls = max(1, min(int(max_calls), MAX_API_CALLS))
    turns = [{"role": "user", "parts": [{"video": str(clip), "fps": fps}, {"text": user}]}]
    trace, step, last_draft = [], 0, None

    def ask():
        text = backend.chat(system, turns)
        # an empty reply (it happens under heavy load) cannot be sent back as a turn: the API rejects
        # empty parts, so keep a stand-in; the next prompt then asks for a valid reply
        turns.append({"role": "model", "parts": [{"text": text or "(empty reply)"}]})
        try:
            return parse_json(text)
        except ValueError:
            return {"_unparseable": text}

    def tool_requests(reply):
        reqs = reply.get("tools")
        if not isinstance(reqs, list):
            reqs = [{"tool": reply["tool"], "args": reply.get("args")}] if "tool" in reply else []
        return [r for r in reqs if isinstance(r, dict) and r.get("tool")][:MAX_TOOLS_PER_BATCH]

    for call in range(max_calls):
        last = call == max_calls - 1
        reply = ask()
        thought = reply.get("thought", "")

        if isinstance(reply.get("final"), dict):
            final = reply["final"]
            issues = validate(final, tools.vocab, tools.registry, ctx, tools.verified_clock)
            trace.append({"step": step, "call": call + 1, "thought": thought, "final": True, "issues": issues})
            step += 1
            if issues and not last:
                log(f"    call {call + 1}: final answer rejected ({len(issues)} problems), sent back")
                turns.append({"role": "user", "parts": [{"text":
                    "Your final answer failed these checks. No more tools"
                    + (": this is your LAST reply" if call == max_calls - 2 else "")
                    + ". Fix them and give the final answer:\n- " + "\n- ".join(issues)}]})
                continue
            log(f"    call {call + 1}: final answer" + (f" ({len(issues)} unresolved problems)" if issues else ""))
            return final, trace, issues, call + 1

        reqs = tool_requests(reply)
        if reqs and call == 0 and not last:
            parts = []
            for r in reqs:
                name, args = r["tool"], r.get("args") or {}
                if name == "validate_draft" and isinstance(args.get("draft"), dict):
                    last_draft = args["draft"]
                result = tools.call(name, args)
                short = (result.get("text") or "").splitlines()[0][:110] if result.get("text") else ""
                log(f"    call {call + 1}: {name}({json.dumps(args)[:80]}) -> {short}")
                trace.append({"step": step, "call": call + 1, "thought": thought, "tool": name, "args": args,
                              "result": result.get("text")})
                step += 1
                parts.append({"text": f"TOOL RESULT ({name}):\n{result.get('text', '')}"})
                parts += [{"image": img} for img in result.get("images", [])]
                if result.get("video"):
                    parts.append({"video": result["video"], "fps": result.get("fps", fps)})
            parts.append({"text": "Those are all your tool results. No more tools. "
                                  'Give your final answer now as {"thought": ..., "final": {...}}.'})
            turns.append({"role": "user", "parts": parts})
            continue

        trace.append({"step": step, "call": call + 1, "invalid_reply": str(reply)[:300]})
        step += 1
        if not last:
            turns.append({"role": "user", "parts": [{"text":
                'That was not a valid reply. No more tools: give your final answer now as '
                '{"thought": ..., "final": {...}}.'}]})

    # out of calls without an accepted final answer: fall back to the last draft the model validated
    final = last_draft or {}
    issues = validate(final, tools.vocab, tools.registry, ctx, tools.verified_clock)
    trace.append({"step": step, "final": True, "forced": True, "from_draft": last_draft is not None, "issues": issues})
    log(f"    call budget used up: {'last validated draft' if last_draft else 'empty answer'} used "
        f"({len(issues)} problems)")
    return final, trace, issues, max_calls
