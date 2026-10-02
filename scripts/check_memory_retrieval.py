"""Replay archived turns through retrieval, without model calls or oracle inputs.

This checks evidence availability, not answer accuracy or clinical efficacy.
Expected values are read only after retrieval has completed.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.core.factual_memory import EvidenceMemory, render_evidence
from agent.core.memory_store import JsonlMemoryStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / "memory").glob("*.jsonl")):
        parser.error("Use a fresh output directory: replay must not load future turns from an earlier attempt")
    memory = EvidenceMemory(JsonlMemoryStore(args.output / "memory"))
    probes = []
    for directory in sorted((args.campaign / "longitudinal/results").iterdir()):
        patient_id = directory.name
        if not (directory / "full").is_dir():
            continue
        for session_path in sorted((directory / "full").glob("session_*/session.json")):
            session = json.loads(session_path.read_text())
            for index, turn in enumerate(session["turns"], 1):
                if session["session_index"] == 11:
                    found = memory.retrieve(patient_id=patient_id, therapist_id="replay",
                                            query=turn["therapist_text"], token_budget=1800)
                    probes.append(dict(turn_id=turn["turn_id"], patient_id=patient_id,
                                       query=turn["therapist_text"], evidence=render_evidence(found),
                                       sources=[r["source_id"] for r in found]))
                memory.record_turn(patient_id=patient_id, therapist_id="replay",
                                   session_id=str(session["session_index"]), turn_index=index,
                                   therapist_text=turn["therapist_text"], patient_text=turn["patient_text"])
    # Evaluation metadata never enters retrieval or its search query.
    by_turn = {p["turn_id"]: p for p in probes}
    assertions = []
    with (args.campaign / "analysis/longitudinal-assertions.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["condition"] != "full" or row["is_primary"] != "True" or row["source_profile_field"]:
                continue
            probe = by_turn[row["turn_id"]]
            expected = json.loads(row["expected_values"])
            available = all(str(value).casefold() in probe["evidence"].casefold() for value in expected)
            assertions.append(dict(assertion_id=row["assertion_id"], expected_values=expected,
                                   evidence_available=available, previously_correct=row["correct"] == "True"))
    report = dict(type="archived dialogue retrieval regression; no model inference",
                  implementation_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] / "agent/core/factual_memory.py").read_bytes()).hexdigest(),
                  retrieval="lexical, raw utterances, no generated facts", token_budget_estimate=1800,
                  evidence_available=sum(a["evidence_available"] for a in assertions),
                  assertions=len(assertions), results=assertions, probes=probes)
    (args.output / "retrieval-replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"results", "probes"}}))
    return 0 if all(a["evidence_available"] for a in assertions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
