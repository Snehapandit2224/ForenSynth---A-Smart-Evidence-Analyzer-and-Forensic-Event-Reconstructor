"""
Per-case people registry, so the same person keeps the same Person_NN label across clips.

Labels follow the generator's video alias format (Person_01..Person_99). They are this agent's
own labels: they will not equal the generator's aliases, which is expected. Entity resolution
(or evaluate.py) maps them.
"""
import hashlib
import json
import random
from pathlib import Path


class PeopleRegistry:
    def __init__(self, path, case_id, seed_file=None, keep_existing=False):
        self.path = Path(path)
        # Folder runs rebuild the registry from scratch, so reruns give the same labels, the same
        # prompts, and therefore hit the model-answer cache. One-clip-at-a-time runs keep it, so a
        # person seen in an earlier clip of the case keeps their label.
        self.people = {}
        if keep_existing and self.path.exists():
            self.people = json.loads(self.path.read_text(encoding="utf-8"))
        elif seed_file:
            # optional: {"Person_14": "grey hoodie, red cap, ..."} to pin your own labels
            seeds = json.loads(Path(seed_file).read_text(encoding="utf-8"))
            self.people = {k: {"appearance": v, "identifiable": True, "seen_in": []} for k, v in seeds.items()}
        self._rng = random.Random(int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16))

    def _new_label(self):
        used = set(self.people)
        if len(used) >= 99:
            raise RuntimeError("more than 99 people in one case")
        while True:
            label = f"Person_{self._rng.randint(1, 99):02d}"
            if label not in used:
                return label

    def resolve(self, match, appearance, identifiable, obs_id):
        """Return the label for this person, creating one if needed."""
        if identifiable and match in self.people and self.people[match].get("identifiable", True):
            label = match
        else:
            label = self._new_label()
            self.people[label] = {"appearance": appearance, "identifiable": bool(identifiable), "seen_in": []}
        self.people[label]["seen_in"].append(obs_id)
        return label

    def forget(self, obs_id):
        """Drop an observation (and anyone seen only in it) before that clip is processed again."""
        base = obs_id.rstrip("abcdefghijklmnopqrstuvwxyz")
        for label in list(self.people):
            seen = [o for o in self.people[label]["seen_in"] if o.rstrip("abcdefghijklmnopqrstuvwxyz") != base]
            if seen or not self.people[label]["seen_in"]:
                self.people[label]["seen_in"] = seen
            else:
                del self.people[label]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.people, indent=2, ensure_ascii=False), encoding="utf-8")
