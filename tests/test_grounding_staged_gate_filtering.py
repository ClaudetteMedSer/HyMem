"""A rejected claim must not quarantine other fully assessed claims."""

import json
import tempfile
import unittest
from pathlib import Path

from hymem.api import HyMem
from hymem.config import HyMemConfig
from hymem.extraction.chunk import extract_chunk
from hymem.extraction import grounding_staged_v1 as staged
from hymem.extraction.grounding_classification_v4 import PREDICATE_ORDER
from hymem.extraction.grounding_staged_gate_v1 import GroundingGateError, ground_triples
from hymem.extraction.producer import (
    Phase1ProducerDeclaration,
    phase1_generation_binding,
    validate_phase1_grounding_policy,
)
from hymem.extraction.triples import Triple


SOURCE = "Atlas uses Redis. Atlas runs on Linux."


def _assessment(state, quote=None):
    if state != "supported":
        return {"state": state, "support": None}
    return {
        "state": state,
        "support": {
            "evidence": [{"source_message_id": None, "region": "owned", "quote": quote}],
            "checks": {
                "attribution_and_roles": {"state": "supported", "evidence_indices": [0]},
                "relation_and_polarity": {"state": "supported", "evidence_indices": [0]},
            },
        },
    }


def _original(batch, states, quotes=None):
    quotes = quotes or {}
    return json.dumps({
        "schema": staged.ORIGINAL_SCHEMA,
        "batch_sha256": batch.batch_sha256,
        "complete": True,
        "originals": [
            {"index": i, "original": _assessment(state, quotes.get(i))}
            for i, state in enumerate(states)
        ],
    })


def _alternatives(batch, supported=None):
    supported = supported or {}
    return json.dumps({
        "schema": staged.ALTERNATIVES_SCHEMA,
        "batch_sha256": batch.classification_batch.batch_sha256,
        "original_response_sha256": batch.original_response_sha256,
        "complete": True,
        "alternatives": [
            {"index": i, "alternatives": {
                predicate: _assessment(
                    "supported" if predicate == supported.get(i) else "not_established",
                    SOURCE.split(". ")[0] if predicate == supported.get(i) else None,
                )
                for predicate in PREDICATE_ORDER
                if predicate != batch.classification_batch.triples[i].predicate
            }}
            for i in batch.negative_indices
        ],
    })


class StagedGroundingFilteringTests(unittest.TestCase):
    def test_validated_unsupported_and_ambiguous_claims_are_omitted(self):
        triples = [
            Triple("Atlas", "uses", "Redis", 1),
            Triple("Atlas", "uses", "MySQL", 1),
            Triple("Atlas", "uses", "Postgres", 1),
        ]
        stages = []

        def invoke(_request, batch, stage, recheck):
            stages.append((stage, recheck))
            if stage == "original":
                return _original(batch, ("supported", "not_established", "ambiguous"),
                                 {0: "Atlas uses Redis"})
            return _alternatives(batch)

        self.assertEqual(ground_triples(
            triples, None, (), SOURCE, invoke,
            diagnostic_grounding_recovery=True,
        ), [triples[0]])
        self.assertEqual(stages, [("original", False), ("alternatives", False)])

    def test_default_policy_keeps_negative_verdict_atomic(self):
        triples = [Triple("Atlas", "uses", "Redis", 1),
                   Triple("Atlas", "uses", "MySQL", 1)]

        def invoke(_request, batch, stage, _recheck):
            return (_original(batch, ("supported", "not_established"),
                              {0: "Atlas uses Redis"}) if stage == "original"
                    else _alternatives(batch))

        with self.assertRaisesRegex(GroundingGateError, "verdict:unsupported"):
            ground_triples(triples, None, (), SOURCE, invoke)

    def test_invalid_evidence_remains_an_atomic_failure(self):
        triples = [Triple("Atlas", "uses", "Redis", 1),
                   Triple("Atlas", "uses", "MySQL", 1)]

        def invoke(_request, batch, stage, recheck):
            self.assertEqual((stage, recheck), ("original", False))
            return _original(batch, ("supported", "not_established"),
                             {0: "Atlas uses an absent database"})

        with self.assertRaisesRegex(GroundingGateError, "contract:evidence_quote_missing"):
            ground_triples(triples, None, (), SOURCE, invoke)

    def test_corrected_claim_is_rechecked_and_rejected_if_support_disappears(self):
        wrong = Triple("Atlas", "prefers", "Redis", 1)
        kept = Triple("Atlas", "runs_on", "Linux", 1)
        stages = []

        def invoke(_request, batch, stage, recheck):
            claims = (batch.classification_batch.triples if stage == "alternatives"
                      else batch.triples)
            stages.append((stage, recheck, tuple(t.predicate for t in claims)))
            if stage == "alternatives":
                return _alternatives(batch, {0: "uses"})
            return _original(batch, ("not_established", "supported"),
                             {1: "Atlas runs on Linux"})

        self.assertEqual(ground_triples(
            [wrong, kept], None, (), SOURCE, invoke,
            diagnostic_grounding_recovery=True,
        ), [kept])
        self.assertEqual([(stage, recheck) for stage, recheck, _ in stages], [
            ("original", False), ("alternatives", False), ("original", True)])
        self.assertEqual(stages[-1][2], ("uses", "runs_on"))

    def test_invalid_batch_evidence_is_reassessed_by_claim(self):
        triples = [Triple("Atlas", "uses", "Redis", 1),
                   Triple("Atlas", "runs_on", "Linux", 1)]
        calls = []
        rejected = []

        def invoke(_request, batch, stage, recheck):
            self.assertEqual((stage, recheck), ("original", False))
            calls.append(len(batch.triples))
            quotes = ({0: "Atlas uses Redis", 1: "Atlas runs on an absent OS"}
                      if len(batch.triples) == 2 else
                      {0: "Atlas uses Redis" if batch.triples[0].object == "Redis"
                       else "Atlas runs on Linux"})
            return _original(batch, ("supported",) * len(batch.triples), quotes)

        self.assertEqual(ground_triples(
            triples, None, (), SOURCE, invoke,
            diagnostic_grounding_recovery=True, rejection_sink=rejected.append,
        ), triples)
        self.assertEqual(calls, [2, 1, 1])
        self.assertEqual(rejected, [])

    def test_all_rejected_claims_publish_no_entity_hints(self):
        class ScriptedClient:
            def complete(self, request):
                if "OMISSION VERIFICATION PASS" in request.system:
                    return json.dumps({"triples": [], "markers": [], "complete": True})
                return json.dumps({
                    "triples": [{
                        "subject": "Atlas", "predicate": "uses", "object": "Redis",
                        "polarity": 1, "subject_type": "service",
                        "subject_properties": {"owner": "Mira"},
                    }],
                    "markers": [], "complete": True,
                })

            def complete_stage(self, _request, batch, stage, _recheck):
                if stage == "original":
                    return _original(batch, ("not_established",))
                return _alternatives(batch)

        result = extract_chunk(
            ScriptedClient(), "Atlas uses Redis.",
            diagnostic_grounding_recovery=True,
        )
        self.assertFalse(result.failed)
        self.assertEqual(result.triples, [])
        self.assertEqual(result.entity_type_hints, {})
        self.assertEqual(result.entity_property_hints, {})
        self.assertEqual(result.diagnostic_grounding_rejections,
                         ("verdict_unsupported",))

    def test_singleton_invalid_evidence_is_explicitly_rejected(self):
        rejected = []
        calls = []

        def invoke(_request, batch, stage, recheck):
            calls.append((stage, recheck))
            return _original(batch, ("supported",),
                             {0: "Atlas uses an absent database"})

        self.assertEqual(ground_triples(
            [Triple("Atlas", "uses", "Redis", 1)], None, (), SOURCE, invoke,
            diagnostic_grounding_recovery=True, rejection_sink=rejected.append,
        ), [])
        self.assertEqual(calls, [("original", False), ("original", False)])
        self.assertEqual(rejected, ["invalid_evidence_quote_missing"])

    def test_structural_assessment_error_stays_atomic_in_diagnostic_mode(self):
        rejected = []

        def invoke(_request, batch, _stage, _recheck):
            payload = json.loads(_original(batch, ("supported",),
                                           {0: "Atlas uses Redis"}))
            payload["originals"][0]["original"]["support"].pop("checks")
            return json.dumps(payload)

        with self.assertRaisesRegex(GroundingGateError, "contract:support_shape"):
            ground_triples(
                [Triple("Atlas", "uses", "Redis", 1)], None, (), SOURCE, invoke,
                diagnostic_grounding_recovery=True, rejection_sink=rejected.append,
            )
        self.assertEqual(rejected, [])


class DiagnosticPolicyBindingTests(unittest.TestCase):
    class DeclaredClient:
        def __init__(self, diagnostic):
            self.diagnostic = diagnostic
            self.calls = 0

        def phase1_producer_declaration(self):
            return Phase1ProducerDeclaration(
                client_id="tests.diagnostic-policy", implementation="sha256:" + "a" * 64,
                model="scripted", endpoint=None,
                effective_request={"diagnostic_grounding_recovery": self.diagnostic},
                retry_policy={"attempts": 1},
            )

        def complete(self, _request):
            self.calls += 1
            raise AssertionError("provider should not be called")

    def test_policy_matches_exact_producer_and_separates_generation_keys(self):
        strict = self.DeclaredClient(False)
        diagnostic = self.DeclaredClient(True)
        strict_generation = phase1_generation_binding("v20", strict)
        diagnostic_generation = phase1_generation_binding("v20", diagnostic)
        self.assertNotEqual(strict_generation["generation_key"],
                            diagnostic_generation["generation_key"])
        validate_phase1_grounding_policy(
            strict, diagnostic_grounding_recovery=False,
            expected_producer=strict_generation["producer"],
        )
        validate_phase1_grounding_policy(
            diagnostic, diagnostic_grounding_recovery=True,
            expected_producer=diagnostic_generation["producer"],
        )
        with self.assertRaisesRegex(ValueError, "disagrees with producer identity"):
            validate_phase1_grounding_policy(
                diagnostic, diagnostic_grounding_recovery=False,
                expected_producer=diagnostic_generation["producer"],
            )
        with self.assertRaisesRegex(ValueError, "disagrees with producer identity"):
            validate_phase1_grounding_policy(
                strict, diagnostic_grounding_recovery=True,
                expected_producer=strict_generation["producer"],
            )

    def test_public_dream_rejects_policy_mismatch_before_provider_call(self):
        with tempfile.TemporaryDirectory() as root:
            diagnostic = self.DeclaredClient(True)
            memory = HyMem(HyMemConfig(root=Path(root)), llm=diagnostic)
            try:
                with self.assertRaisesRegex(ValueError, "disagrees with producer identity"):
                    memory.dream()
                self.assertEqual(diagnostic.calls, 0)
            finally:
                memory.close()

            strict = self.DeclaredClient(False)
            memory = HyMem(HyMemConfig(root=Path(root)), llm=strict)
            try:
                with self.assertRaisesRegex(ValueError, "disagrees with producer identity"):
                    memory.dream(diagnostic_grounding_recovery=True)
                self.assertEqual(strict.calls, 0)
            finally:
                memory.close()


if __name__ == "__main__":
    unittest.main()
