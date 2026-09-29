"""Derive a fresh, inactive classification diagnostic from pinned sources.

Only prepare() writes, and only to a new private output directory. This module
does not import generated code, use a network transport, or launch a run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


FAMILY = "luna-classification-probe"
OLD_FAMILY = "luna-semantic-probe"
INVENTORY = Path("/private/tmp/hymem-classification-root-ZP2zuU/map.json")
CANDIDATE = INVENTORY.with_name("candidate")
ACCEPTED_CANDIDATE = Path("/home/atta/.hymem-luna-semantic-probe-30jkk7eg/candidate")
ACCEPTED_MAP = ACCEPTED_CANDIDATE.with_name("candidate-source-map.json")
INVENTORY_SHA = "c33953aab2707eb5f9ff650f64698cebbcd5b576cc24c33f3d0cd26b3c9708f5"
EXTRACTION_IDENTITY = "hymem-extraction-contract-sha256-v1:eed0f35aa3edc354e4f482d287e8d3087b86e4342ef3f745f0095229b9a875f6"
OLD_IDENTITY = "hymem-extraction-contract-sha256-v1:f349e2fa14d1778bc556869d346ca44183c025e779a919b3f15e82f7c3d78d46"
OLD_INVENTORY_SHA = "11ca4cdbba18e4b7e4b56d444062e39b789820b2f0055a32898bbd0b3a1e4664"
OLD_CANARY_SHA = "3d132415573cb46c2b15a69a56776ea69b3410b350a8fdb8ba6a1f8cc85658af"
OLD_CORE_SHA = "3ccb7ec12f93f8502fe8ed1448a071ac977dd334d17821f661913d634c27b334"
OLD_HOST_SHA = "7e190561958b126593a3e652ec2de35fb4cef27d4e5863284a75636abe531b07"
OLD_RUN_SHA = "60fc7fca900ef8320a410af2f5f01415f973f01907cf7c0456c0b4936611030a"
OLD_READER_SHA = "68f02223a561dd31a1a0417d8972ddafab7cbd26fc15500a9a2efc1b0cb9d355"
BUILDER_SHA = "cb0f904bf67d12c1099150adbcee68dbdbe78c63069337d77c14c296bcb3b8ca"
ADAPTER_SHA = "bf2b0e52d9e5ae8b6c5e6c01822f17afd0f21384006fa69fa6ab03bb8aed8511"
CLASSIFICATION_SHA = "7c62c6e58305a5b7be256825a52a8cc4b9119888011bf18fe267b18dccab2c06"
GATE_SHA = "0c676c9e10f9dfb9fa005be4ec7b75c799d4e69e37d61ae0953a52e86ed59081"
REPLAY_SHA = "38806d0c42c60c1af89a581d96be5657ad5a4c2a3517fd53c28f354f9b39e6ea"
INPUT_SHA = {
    "tools/diagnostics/luna_semantic_probe.py": OLD_CORE_SHA,
    "tools/diagnostics/luna_semantic_cases.py": "8876e94e6c5d3284d4d361f0238e0aef507288e2702b5df77ee706fd4c712d8a",
    "tools/diagnostics/luna_semantic_candidate.py": "a10ee5c5a1ba4f6a2694a88570a399c5fe081db02f0cb0ecfa5b92e5db06d7b2",
    "tools/diagnostics/luna_semantic_candidate_v2.py": "ed0c5c006d316fb0c762a2df6977313403e45fafdb914ea3c4f2f675a0165e3f",
    "tools/diagnostics/luna_classification_candidate.py": BUILDER_SHA,
    "benchmarks/luna_semantic_canary.py": OLD_CANARY_SHA,
    "benchmarks/luna_semantic_stage_accounting.py": "4c654599a979e51aeb9f0985b091cd8ef38a639424f197c412da45e1f3270d30",
    "benchmarks/codex_subscription_warm_v3.py": "0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d",
    "benchmarks/codex_subscription_warm_v2.py": "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593",
    "benchmarks/codex_subscription_concurrent_v2.py": "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0",
    "benchmarks/codex_subscription.py": "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491",
    "benchmarks/codex_subscription_classification_v1.py": ADAPTER_SHA,
    "hymem/extraction/grounding.py": "dd1a49b56abf569b4476a0b735e67e72a86723bf88e3a4739998aceeabe19c18",
    "hymem/extraction/grounding_gate.py": "bb79e1b0baa1a16032532fb73ec448a7dd3dcab94fbf87f69b6e7931489f03f8",
    "hymem/extraction/grounding_v2.py": "377a688caf183f3645be246b77a445bd94d053def00fff87589061b4b2fc31ec",
    "hymem/extraction/grounding_classification_v1.py": CLASSIFICATION_SHA,
    "hymem/extraction/grounding_classification_gate_v1.py": GATE_SHA,
    "tools/diagnostics/luna_semantic_probe_host.py": OLD_HOST_SHA,
    "tools/diagnostics/luna_semantic_probe_run.py": OLD_RUN_SHA,
    "tools/diagnostics/luna_semantic_probe_progress.py": OLD_READER_SHA,
    "tools/diagnostics/luna_semantic_probe_adapter_v2.py": "e6ac48313364367756afbbaa7720e9d65047b12590237d5974f8324b188bf504",
    "tools/diagnostics/luna_semantic_verdict_replay_root.py": REPLAY_SHA,
}
CODE = tuple(k for k in INPUT_SHA if k not in {
    "tools/diagnostics/luna_semantic_probe_adapter_v2.py",
    "tools/diagnostics/luna_semantic_verdict_replay_root.py"})
HEX = re.compile(r"[0-9a-f]{64}\Z")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source(path: Path, expected: str) -> bytes:
    if (not path.is_absolute() or path != path.resolve() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode)):
        raise ValueError("source_path_invalid")
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("source_pin_invalid:" + path.name)
    return raw


def replace(raw: bytes, old: str, new: str, count: int = 1) -> bytes:
    before, after = old.encode(), new.encode()
    if raw.count(before) != count:
        raise ValueError(f"replacement_count_invalid:{old[:80]}:{raw.count(before)}:{count}")
    return raw.replace(before, after)


def family(raw: bytes, count: int) -> bytes:
    return replace(raw, OLD_FAMILY, FAMILY, count)


def validate_candidate() -> dict[str, str]:
    stamp = json.loads(source(INVENTORY, INVENTORY_SHA))
    mapping = stamp.get("source_sha256")
    if type(mapping) is not dict or len(mapping) != 513:
        raise ValueError("candidate_map_invalid")
    if (not CANDIDATE.is_absolute() or CANDIDATE != CANDIDATE.resolve()
            or CANDIDATE.is_symlink() or not CANDIDATE.is_dir()):
        raise ValueError("candidate_path_invalid")
    actual = {}
    for path in CANDIDATE.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("candidate_path_invalid")
        if path.is_file():
            actual[path.relative_to(CANDIDATE).as_posix()] = sha(path.read_bytes())
    if (actual != mapping or mapping.get("hymem/extraction/grounding_classification_v1.py") != CLASSIFICATION_SHA
            or mapping.get("hymem/extraction/grounding_classification_gate_v1.py") != GATE_SHA
            or any(type(k) is not str or type(v) is not str or HEX.fullmatch(v) is None
                   or Path(k).is_absolute() or ".." in Path(k).parts for k, v in mapping.items())):
        raise ValueError("candidate_inventory_invalid")
    return mapping


def write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def derive_core(raw: bytes, canary_sha: str, stage_sha: str) -> bytes:
    raw = family(raw, 1)
    for old, new in ((OLD_CANARY_SHA, canary_sha), (OLD_INVENTORY_SHA, INVENTORY_SHA),
                     (INPUT_SHA["benchmarks/luna_semantic_stage_accounting.py"], stage_sha)):
        raw = replace(raw, old, new)
    raw = replace(raw, 'len(mapping) == 510', 'len(mapping) == 513')
    raw = replace(raw, '    fixture = cases_module.cases()\n',
                  '    fixture = cases_module.cases()\n'
                  '    from hymem.extraction import grounding as original_grounding\n'
                  '    from hymem.extraction.grounding_classification_gate_v1 import _v2_source\n')
    raw = replace(raw, 'type(source) is grounding_module.GroundingSource',
                  'type(source) is original_grounding.GroundingSource')
    raw = replace(raw, '    stage_module.verify_candidate(candidate)\n',
                  '    require(all(type(_v2_source(source)) is grounding_module.GroundingSource\n'
                  '                for case in fixture for source in case.sources), "fixture_conversion_invalid")\n'
                  '    stage_module.verify_candidate(candidate)\n')
    raw = replace(raw, 'candidate.resolve() / "hymem/extraction/grounding.py"',
                  'candidate.resolve() / "hymem/extraction/grounding_classification_v1.py"')
    raw = replace(raw, '    request, batch = grounding.build_grounding_request(case.triples, case.sources)\n',
                  '    from hymem.extraction.grounding_classification_gate_v1 import _v2_source\n'
                  '    trusted_sources = tuple(_v2_source(source) for source in case.sources)\n'
                  '    request, batch = grounding.build_grounding_request(case.triples, trusted_sources)\n')
    raw = replace(raw, '"batch_sha256": batch.batch_sha256})',
                  '"batch_sha256": batch.batch_sha256, "batch": batch.canonical_json})', 1)
    raw = replace(raw, '    raw = client.complete(request)\n',
                  '    raw = client.complete_grounding(request, batch)\n', 1)
    raw = replace(raw, 'grounding.build_grounding_request(corrected, case.sources)',
                  'grounding.build_grounding_request(corrected, trusted_sources)')
    raw = replace(raw, '"batch_sha256": batch2.batch_sha256})',
                  '"batch_sha256": batch2.batch_sha256, "batch": batch2.canonical_json})')
    raw = replace(raw, '        raw2 = client.complete(request2)\n',
                  '        raw2 = client.complete_grounding(request2, batch2)\n')
    start = raw.index(b'    def complete(self, request):\n', raw.index(b'class HybridReplayClient:'))
    end = raw.index(b'\n\ndef run_hybrid(', start)
    raw = raw[:start] + HYBRID_METHOD.encode() + raw[end:]
    raw = replace(raw, 'Path(warm.__file__).name == "codex_subscription_warm_v3.py"',
                  'Path(warm.__file__).name == "codex_subscription_warm_v3.py"')
    raw = replace(raw, '    factory = client_factory or (lambda key, cap, budget: warm.WarmSubscriptionClient(\n',
                  '    from benchmarks.codex_subscription_classification_v1 import ClassificationSubscriptionClient\n'
                  '    factory = client_factory or (lambda key, cap, budget: ClassificationSubscriptionClient(\n')
    return raw


HYBRID_METHOD = '''    def complete(self, request):
        require(self.replayed < 8, "unexpected_extra_ordinary")
        expected, response = self.retained[self.replayed]
        require(canonical(asdict(request)) == canonical(expected),
                "ordinary_request_mismatch")
        self.record({"phase": "ordinary_replay", "index": self.replayed,
                     "request_sha256": digest(canonical(expected)),
                     "response_sha256": digest(response.encode("utf-8"))})
        self.replayed += 1
        return response

    def complete_grounding(self, request, batch):
        from hymem.extraction.grounding_classification_v1 import validate_request
        validate_request(request, batch)
        require(self.new_calls < 3 and len(batch.triples) == 1,
                "grounding_call_limit")
        wanted_id = self.source_ids[0] if self.new_calls == 0 else self.source_ids[1]
        require(batch.triples[0].source_message_id == wanted_id,
                "grounding_order_invalid")
        phase = ("table_initial" if self.new_calls == 0 else
                 "prose_initial" if self.new_calls == 1 else "prose_recheck")
        expected_replay = 4 if self.new_calls == 0 else 8
        require(self.replayed == expected_replay, "hybrid_sequence_invalid")
        self.record({"phase": phase + "_before_dispatch", "request": asdict(request),
                     "batch": batch.canonical_json,
                     "batch_sha256": batch.batch_sha256,
                     "request_sha256": digest(canonical(asdict(request)))})
        answer = self.paid.complete_grounding(request, batch)
        self.record({"phase": phase + "_returned", "response": answer})
        self.new_calls += 1
        self.grounding_phases.append(phase)
        return answer
'''


def derive_canary(raw: bytes) -> bytes:
    raw = replace(raw, 'luna-semantic-canary-v1', 'luna-classification-canary-v1')
    raw = replace(raw, OLD_IDENTITY, EXTRACTION_IDENTITY)
    begin = raw.index(b'def _is_grounding(')
    end = raw.index(b'\n\ndef validate_report(', begin)
    raw = raw[:begin] + CANARY_TRACE.encode() + raw[end:]
    raw = replace(raw, '    recording = canary._RecordingClient(client)\n',
                  '    recording = _RecordingClient(client)\n')
    raw = replace(raw, '        for request, response in ground:\n',
                  '        for request, response in ground:\n')
    begin = raw.index(b'            wire = json.loads(request.user)\n',
                      raw.index(b'    try:\n        for request, response in ground:'))
    end = raw.index(b'        if correction_pending:\n', begin)
    raw = raw[:begin] + CANARY_LOOP.encode() + raw[end:]
    raw = replace(raw, '    ground = [(r, v) for r, v in pairs if _is_grounding(r)]\n',
                  '    ground = [(r, v) for r, v in pairs if id(r) in recording.batches]\n')
    raw = replace(raw, '    ordinary = [(r, v) for r, v in pairs if not _is_grounding(r)]\n',
                  '    ordinary = [(r, v) for r, v in pairs if id(r) not in recording.batches]\n')
    raw = replace(raw, '"responses": [r for _, r in recording.responses]})',
                  '"responses": [r for _, r in recording.responses],\n'
                  '                      "trusted_batches": [recording.batches.get(id(r)).canonical_json\n'
                  '                                          if id(r) in recording.batches else None\n'
                  '                                          for r in recording.requests]})')
    return raw


CANARY_TRACE = '''class _RecordingClient:
    """Record ordinary requests and exact trusted classification batches."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.requests = []
        self.responses = []
        self.batches = {}
        self.provider_output_truncations = 0

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def complete(self, request):
        self.requests.append(request)
        try:
            answer = self.delegate.complete(request)
        except BaseException as exc:
            from hymem.extraction.llm import LLMOutputTruncatedError
            if isinstance(exc, LLMOutputTruncatedError):
                self.provider_output_truncations += 1
            raise
        self.responses.append((request, answer))
        return answer

    def complete_grounding(self, request, batch):
        self.requests.append(request)
        self.batches[id(request)] = batch
        try:
            answer = self.delegate.complete_grounding(request, batch)
        except BaseException as exc:
            from hymem.extraction.llm import LLMOutputTruncatedError
            if isinstance(exc, LLMOutputTruncatedError):
                self.provider_output_truncations += 1
            raise
        self.responses.append((request, answer))
        return answer


def _safe_grounding_trace(request, batch, response, expected, *, recheck,
                          source_payloads, ordinary_payloads):
    """Validate source, context, trusted original and selected verdict."""
    from hymem.extraction import grounding_classification_v1 as grounding
    from hymem.extraction.grounding_classification_gate_v1 import _v2_source
    from hymem.extraction.grounding_gate import _source

    grounding.validate_request(request, batch)
    if len(batch.triples) != 1 or len(batch.sources) != 1:
        raise ValueError("grounding_binding_invalid")
    candidate = batch.triples[0]
    source = batch.sources[0]
    index = next((i for i, e in enumerate(expected)
                  if (candidate.subject, candidate.object, candidate.polarity,
                      candidate.source_message_id) == (e[0], e[3], e[5], e[6])), None)
    if index is None or candidate.source_message_id != source.source_message_id:
        raise ValueError("grounding_candidate_invalid")
    e = expected[index]
    if any(getattr(candidate, field) is not None for field in _OPTIONAL):
        raise ValueError("grounding_qualifier_invalid")
    if recheck and candidate.predicate != e[2]:
        raise ValueError("recheck_candidate_invalid")
    if (source.content not in source_payloads[e[6]]["content"] or
            any(getattr(source, field) != source_payloads[e[6]].get(field)
                for field in ("source_role", "source_peer_id", "source_created_at"))):
        raise ValueError("grounding_source_invalid")
    matching = [payload for payload in ordinary_payloads
                if payload.get("source_message_id") == e[6]
                and payload.get("content") == source.content]
    expected_sources = []
    for payload in matching:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"))
        expected_sources.append(_v2_source(_source((e[6], encoded), ())))
    if source not in expected_sources:
        raise ValueError("grounding_context_invalid")
    review = grounding.parse_grounding_response(response, batch,
                                                allow_corrections=not recheck)
    if len(review.verdicts) != 1:
        raise ValueError("grounding_verdict_invalid")
    verdict = review.verdicts[0]
    if recheck:
        if verdict.status != "supported" or verdict.predicate != e[2]:
            raise ValueError("recheck_verdict_invalid")
    elif verdict.status == "replace_predicate":
        if candidate.predicate == e[2] or verdict.predicate != e[2]:
            raise ValueError("correction_invalid")
    elif (verdict.status != "supported" or candidate.predicate != e[2]
          or verdict.predicate != e[2]):
        raise ValueError("unsupported_raw_claim")
    return {"claim_index": index, "batch_sha256": batch.batch_sha256,
            "status": verdict.status, "recheck": recheck,
            "candidate_predicate_expected": candidate.predicate == e[2],
            "owned_source_bound": True}
'''


CANARY_LOOP = '''            batch = recording.batches[id(request)]
            index = next(i for i, e in enumerate(expected)
                         if batch.triples[0].source_message_id == e[6])
            recheck = index in correction_pending
            if index in initial_seen and not recheck:
                raise ValueError("repeated_initial")
            item = _safe_grounding_trace(request, batch, response, expected,
                                         recheck=recheck, source_payloads=source_payloads,
                                         ordinary_payloads=ordinary_payloads)
            trace.append(item)
            if recheck:
                correction_pending.remove(index)
            else:
                initial_seen.add(index)
                if item["status"] == "replace_predicate":
                    correction_pending.add(index)
'''


def derive_stage(raw: bytes, inventory: dict[str, str]) -> bytes:
    raw = replace(raw, '"hymem/extraction/chunk.py": "0dc650c244d3aea75b79308ada28593442382ff9f52ad88bd086982e76e6690b"',
                  '"hymem/extraction/chunk.py": "' + inventory["hymem/extraction/chunk.py"] + '"')
    raw = replace(raw, '"hymem/extraction/grounding_gate.py": "bb79e1b0baa1a16032532fb73ec448a7dd3dcab94fbf87f69b6e7931489f03f8"',
                  '"hymem/extraction/grounding_classification_gate_v1.py": "' + GATE_SHA + '"')
    begin = raw.index(b'    gate = "hymem/extraction/grounding_gate.py"\n')
    end = raw.index(b'    if (chunk, "single_attempt", 2975) in sites:\n', begin)
    raw = raw[:begin] + (
        '    if (chunk, "grounding_call", 3215) in sites:\n'
        '        return "grounding_recheck"\n'
        '    if (chunk, "grounding_call", 3216) in sites:\n'
        '        return "grounding_initial"\n').encode() + raw[end:]
    for old, new in ((3172, 3173), (3360, 3363), (3305, 3308), (3239, 3242)):
        raw = replace(raw, f'(chunk, "{"attempt" if old == 3172 else "recover" if old in (3360, 3305) else "verify_nonempty"}", {old})',
                      f'(chunk, "{"attempt" if old == 3172 else "recover" if old in (3360, 3305) else "verify_nonempty"}", {new})')
    raw = replace(raw, '(chunk, "single_attempt", 2975)',
                  '(chunk, "single_attempt", 2976)')
    raw = replace(raw, '    def complete(self, request):\n        stage = classify_stack(self.ledger.candidate)\n',
                  '    def complete(self, request):\n        return self._invoke(request, None)\n\n'
                  '    def complete_grounding(self, request, batch):\n'
                  '        return self._invoke(request, batch)\n\n'
                  '    def _invoke(self, request, batch):\n'
                  '        stage = classify_stack(self.ledger.candidate)\n')
    raw = replace(raw, '            result = self.delegate.complete(request)\n',
                  '            result = (self.delegate.complete(request) if batch is None else\n'
                  '                      self.delegate.complete_grounding(request, batch))\n')
    return raw


def derive_transport(raw: bytes) -> bytes:
    return replace(raw,
        'Path(__file__).resolve().parents[1] / "hymem/extraction/grounding_classification_v1.py"',
        'Path(__file__).resolve().parents[2] / "candidate/hymem/extraction/grounding_classification_v1.py"')


def derive_host(raw: bytes, sources: dict[str, bytes], core_sha: str, canary_sha: str,
                stage_sha: str, adapter_sha: str) -> bytes:
    raw = family(raw, 4)
    for old, new in ((OLD_CORE_SHA, core_sha), (OLD_CANARY_SHA, canary_sha),
                     (OLD_INVENTORY_SHA, INVENTORY_SHA), (OLD_IDENTITY, EXTRACTION_IDENTITY),
                     (INPUT_SHA["benchmarks/luna_semantic_stage_accounting.py"], stage_sha)):
        raw = replace(raw, old, new)
    anchor = "    'tools/diagnostics/luna_semantic_candidate.py': '" + INPUT_SHA["tools/diagnostics/luna_semantic_candidate.py"] + "',"
    additions = [anchor]
    for name in ("tools/diagnostics/luna_semantic_candidate_v2.py",
                 "tools/diagnostics/luna_classification_candidate.py",
                 "benchmarks/codex_subscription_classification_v1.py",
                 "hymem/extraction/grounding_v2.py",
                 "hymem/extraction/grounding_classification_v1.py",
                 "hymem/extraction/grounding_classification_gate_v1.py"):
        digest = adapter_sha if name.endswith("codex_subscription_classification_v1.py") else INPUT_SHA[name]
        additions.append(f"    '{name}': '{digest}',")
    raw = replace(raw, anchor, "\n".join(additions))
    raw = replace(raw, "source = root / 'code/tools/diagnostics/luna_semantic_candidate.py'",
                  "source = root / 'code/tools/diagnostics/luna_classification_candidate.py'")
    raw = replace(raw, "builder.prepare(OLD_CANDIDATE, OLD_MAP, root / 'candidate',\n"
                       "                            root / 'candidate-source-map.json', root / 'code')",
                  "builder.prepare(OLD_CANDIDATE, OLD_MAP, ACCEPTED_CANDIDATE, ACCEPTED_MAP,\n"
                  "                            root / 'candidate', root / 'candidate-source-map.json', root / 'code')")
    raw = replace(raw, "OLD_MAP = OBSERVED / 'headless-grounding-source-map.json'",
                  "OLD_MAP = OBSERVED / 'headless-grounding-source-map.json'\n"
                  "ACCEPTED_CANDIDATE = Path('/home/atta/.hymem-luna-semantic-probe-30jkk7eg/candidate')\n"
                  "ACCEPTED_MAP = ACCEPTED_CANDIDATE.with_name('candidate-source-map.json')")
    raw = replace(raw, "proof['files'] == 510", "proof['files'] == 513")
    raw = replace(raw, "len(mapping) == 510", "len(mapping) == 513")
    raw = replace(raw, "'original_map': str(OLD_MAP),", "'original_map': str(OLD_MAP),\n"
                  "            'accepted_candidate': str(ACCEPTED_CANDIDATE),\n"
                  "            'accepted_map': str(ACCEPTED_MAP),")
    return raw


def derive_run(raw: bytes, core_sha: str, host_sha: str) -> bytes:
    raw = family(raw, 5)
    raw = replace(raw, OLD_CORE_SHA, core_sha)
    raw = replace(raw, "proof['candidate_files'] != 510", "proof['candidate_files'] != 513")
    raw = replace(raw, "from hymem.extraction import grounding, chunk",
                  "from hymem.extraction import grounding_classification_v1 as grounding, chunk")
    raw = replace(raw, "    from benchmarks import codex_subscription_warm_v3 as warm\n    concurrent = warm.concurrent",
                  "    from benchmarks import codex_subscription_classification_v1 as adapter\n"
                  "    warm = adapter.warm\n    concurrent = warm.concurrent")
    raw = replace(raw, "    class TrackingSession(warm.WarmSession):", "    class TrackingSession(warm.WarmSession):")
    raw = replace(raw, "return warm.WarmSubscriptionClient(str(host.BINARY), budget, key,",
                  "return adapter.ClassificationSubscriptionClient(str(host.BINARY), budget, key,")
    raw = replace(raw,
                  "    host, core, cases, grounding, canary, chunk, semantic, stage, warm, concurrent = loaded\n",
                  "    host, core, cases, grounding, canary, chunk, semantic, stage, warm, concurrent = loaded\n"
                  "    from benchmarks import codex_subscription_classification_v1 as adapter\n"
                  "    if adapter.warm is not warm:\n"
                  "        raise core.ProbeStop('transport_import_drift')\n")
    return raw


def derive_reader(raw: bytes) -> bytes:
    raw = family(raw, 6)
    raw = replace(raw, "from benchmarks import codex_subscription_warm_v3 as warm",
                  "from benchmarks.codex_subscription_classification_v1 import warm")
    return raw


def derive_startup(raw: bytes, host_sha: str, run_sha: str, reader_sha: str) -> bytes:
    raw = family(raw, 12)
    for old, new in ((OLD_HOST_SHA, host_sha), (OLD_RUN_SHA, run_sha),
                     (OLD_READER_SHA, reader_sha)):
        raw = replace(raw, old, new)
    return raw


def derive_replay(raw: bytes) -> bytes:
    raw = replace(raw, 'luna-semantic-probe-v1', FAMILY + '-v1')
    raw = replace(raw, 'luna-semantic-verdict-root-replay-v1',
                  FAMILY + '-verdict-root-replay-v1')
    raw = replace(raw, 'pending=(phase[:-len(\'_before_dispatch\')],event[\'request\'])',
                  'pending=(phase[:-len(\'_before_dispatch\')],event[\'request\'],event[\'batch\'],event[\'batch_sha256\'])')
    raw = replace(raw, 'self.pairs.append((pending[1],event[\'response\']))',
                  'self.pairs.append((pending[1],pending[2],pending[3],event[\'response\']))')
    begin = raw.index(b'        def complete(self,request):\n')
    end = raw.index(b'        def all_consumed(self):', begin)
    raw = raw[:begin] + REPLAY_COMPLETE.encode() + raw[end:]
    return raw


REPLAY_COMPLETE = '''        def complete(self,request):
            raise AssertionError('unexpected_ordinary_dispatch')
        def complete_grounding(self,request,batch):
            expected,trusted,expected_sha,raw=self.pairs[self.observed_turns]
            assert core.canonical(asdict(request))==core.canonical(expected)
            assert type(trusted) is str and batch.canonical_json==trusted
            assert batch.batch_sha256==expected_sha
            grounding.validate_request(request,batch)
            self.observed_turns+=1
            self.observed_tokens+=1
            return raw
'''


def prepare(repo: Path, target: Path) -> dict:
    if (not repo.is_absolute() or repo != repo.resolve() or not repo.is_dir()
            or repo.is_symlink() or not target.is_absolute() or
            target.parent != target.parent.resolve() or target.exists() or
            target.is_symlink() or not target.parent.is_dir() or
            target.is_relative_to(repo) or repo.is_relative_to(target) or
            target.is_relative_to(INVENTORY.parent) or INVENTORY.parent.is_relative_to(target)):
        raise ValueError("output_boundary_invalid")
    sources = {name: source(repo / name, digest) for name, digest in INPUT_SHA.items()}
    validate_candidate()
    generated = {name: sources[name] for name in CODE}
    # Each changed code copy is derived below from its exact pinned old bytes.
    canary = derive_canary(sources["benchmarks/luna_semantic_canary.py"])
    stage = derive_stage(sources["benchmarks/luna_semantic_stage_accounting.py"],
                         json.loads(INVENTORY.read_bytes())["source_sha256"])
    generated["benchmarks/luna_semantic_canary.py"] = canary
    generated["benchmarks/luna_semantic_stage_accounting.py"] = stage
    core = derive_core(sources["tools/diagnostics/luna_semantic_probe.py"], sha(canary), sha(stage))
    generated["tools/diagnostics/luna_semantic_probe.py"] = core
    adapter = derive_transport(sources["benchmarks/codex_subscription_classification_v1.py"])
    generated["benchmarks/codex_subscription_classification_v1.py"] = adapter
    host = derive_host(sources["tools/diagnostics/luna_semantic_probe_host.py"],
                       sources, sha(core), sha(canary), sha(stage), sha(adapter))
    generated["tools/diagnostics/luna_semantic_probe_host.py"] = host
    run = derive_run(sources["tools/diagnostics/luna_semantic_probe_run.py"], sha(core), sha(host))
    generated["tools/diagnostics/luna_semantic_probe_run.py"] = run
    reader = derive_reader(sources["tools/diagnostics/luna_semantic_probe_progress.py"])
    generated["tools/diagnostics/luna_semantic_probe_progress.py"] = reader
    startup = derive_startup(sources["tools/diagnostics/luna_semantic_probe_adapter_v2.py"],
                             sha(host), sha(run), sha(reader))
    replay = derive_replay(sources["tools/diagnostics/luna_semantic_verdict_replay_root.py"])
    output_sha = {"code/" + name: sha(raw) for name, raw in generated.items()}
    output_sha.update({"adapter-v2.py": sha(startup), "verdict-replay.py": sha(replay)})
    receipt = {"schema": FAMILY + "-bundle-v1", "candidate_inventory_sha256": INVENTORY_SHA,
               "candidate_files": 513, "extraction_identity": EXTRACTION_IDENTITY,
               "input_sha256": dict(sorted(INPUT_SHA.items())),
               "output_sha256": dict(sorted(output_sha.items())),
               "unchanged_code": sorted(name for name in generated if generated[name] == sources[name]),
               "model_calls": 0, "launched": False}
    target.mkdir(mode=0o700)
    for name, raw in generated.items():
        write(target / "code" / name, raw)
    write(target / "adapter-v2.py", startup)
    write(target / "verdict-replay.py", replay)
    write(target / "derivation-receipt.json", (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    return {"target": str(target), "code_files": len(generated), "receipt_sha256": sha((target / "derivation-receipt.json").read_bytes()),
            "output_sha256": output_sha, "model_calls": 0, "launched": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.repo, args.target), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
