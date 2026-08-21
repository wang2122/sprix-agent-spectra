"""Tamper-evident JSONL evidence ledger for evaluation trajectories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import SelectionDecision, TrialOutcome

GENESIS_HASH = "0" * 64


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class EvidenceRecord:
    sequence: int
    previous_hash: str
    outcome: dict[str, object]
    decision: dict[str, object] | None
    record_hash: str

    def unsigned_payload(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "previous_hash": self.previous_hash,
            "outcome": self.outcome,
            "decision": self.decision,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.unsigned_payload(), "record_hash": self.record_hash}


class EvidenceLedger:
    """Append-only hash chain for outcome and selection evidence.

    A hash chain detects accidental or post-export modification. It does not
    authenticate the evaluator; production deployments should additionally
    sign the final root hash with an organization-controlled key.
    """

    def __init__(self) -> None:
        self.records: list[EvidenceRecord] = []

    @property
    def root_hash(self) -> str:
        return self.records[-1].record_hash if self.records else GENESIS_HASH

    def append(self, outcome: TrialOutcome, decision: SelectionDecision | None = None) -> EvidenceRecord:
        sequence = len(self.records)
        payload = {
            "sequence": sequence,
            "previous_hash": self.root_hash,
            "outcome": asdict(outcome),
            "decision": asdict(decision) if decision is not None else None,
        }
        record_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        record = EvidenceRecord(record_hash=record_hash, **payload)
        self.records.append(record)
        return record

    def verify(self) -> tuple[bool, str]:
        previous = GENESIS_HASH
        for expected_sequence, record in enumerate(self.records):
            if record.sequence != expected_sequence:
                return False, f"sequence mismatch at record {expected_sequence}"
            if record.previous_hash != previous:
                return False, f"previous hash mismatch at record {expected_sequence}"
            expected = hashlib.sha256(_canonical_json(record.unsigned_payload()).encode("utf-8")).hexdigest()
            if record.record_hash != expected:
                return False, f"record hash mismatch at record {expected_sequence}"
            previous = record.record_hash
        return True, "ledger verified"

    def to_jsonl(self) -> str:
        return "\n".join(_canonical_json(record.to_dict()) for record in self.records) + ("\n" if self.records else "")

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_jsonl(), encoding="utf-8")

    @classmethod
    def from_jsonl(cls, content: str, *, verify: bool = True) -> EvidenceLedger:
        ledger = cls()
        for line in content.splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            expected_fields = {"sequence", "previous_hash", "outcome", "decision", "record_hash"}
            if set(data) != expected_fields:
                raise ValueError("unexpected or missing evidence record fields")
            ledger.records.append(
                EvidenceRecord(
                    sequence=int(data["sequence"]),
                    previous_hash=str(data["previous_hash"]),
                    outcome=dict(data["outcome"]),
                    decision=dict(data["decision"]) if data["decision"] is not None else None,
                    record_hash=str(data["record_hash"]),
                )
            )
        if verify:
            valid, reason = ledger.verify()
            if not valid:
                raise ValueError(reason)
        return ledger

    @classmethod
    def read(cls, path: str | Path, *, verify: bool = True) -> EvidenceLedger:
        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"), verify=verify)
