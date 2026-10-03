"""Archive-only canary commitment validation, never live execution admission.

The v17 template was captured from the genuine synthetic reconstructed-R3
fixture d9fd6ededba452c29ef039f785c3fb377c6400e18ea3023ebf4f8a328779e6aa.
Do not reconstruct its extraction hash from current code or relabel its version.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import re

try:
    from .extraction_canary import _integrity, _strict_equal, _validate_report_with_policy
except ImportError:  # direct benchmark scripts
    from extraction_canary import _integrity, _strict_equal, _validate_report_with_policy

# Literal non-runtime policy; independent of today's fixture builders/hashes.
_V17_POLICY = json.loads(r'''{
  "clean_empty_recovery_policy_version": "hymem-terminal-clean-empty-verification-v3",
  "expected_claims": [
    {
      "object": "Fly.io",
      "object_properties": {},
      "object_type": "platform",
      "polarity": 1,
      "predicate": "deploys_to",
      "source_message_id": 9271604311,
      "subject": "HyMem Canary Relay",
      "subject_properties": {},
      "subject_type": "service",
      "temporal_scope": null,
      "value_numeric": null,
      "value_text": null,
      "value_unit": null
    },
    {
      "object": "PostgreSQL",
      "object_properties": {},
      "object_type": "database",
      "polarity": 1,
      "predicate": "prefers",
      "source_message_id": 9271604312,
      "subject": "Avery Boundary Canary",
      "subject_properties": {},
      "subject_type": "person",
      "temporal_scope": null,
      "value_numeric": null,
      "value_text": null,
      "value_unit": null
    }
  ],
  "expected_prepartition_leaves": 4,
  "fenced_code_control_sha256": "c11d7b0dbbb2492184ec94c43894ca2fc9ee3d2991b2878f296fb834da6db274",
  "fixture_sha256": "3fedadfbcdf35a013f8f61c5ea3f6ccc3e150aca0d025d8597327704551e94de",
  "fixture_version": "hymem-phase1-context-paths-four-leaf-v7",
  "list_control_sha256": "e8f93e21ed6b8ea2df448e4ea2b3c7b09122838a2da3999f7c117ff3257a53e5",
  "max_completion_calls": 24,
  "max_provider_attempts": 72,
  "minimum_pass_completion_calls": 8,
  "normal_execution_path": {
    "empty_verification_requests": 2,
    "fenced_code_control_atomic_requests": 2,
    "fenced_code_control_probe_atomic": true,
    "list_control_atomic_requests": 2,
    "list_control_probe_atomic": true,
    "omission_verification_requests": 2,
    "parsed_source_records": 8,
    "primary_requests": 4,
    "prose_claim_exact_context_emissions": 1,
    "prose_claim_exact_context_requests": 2,
    "prose_claim_requests": 2,
    "prose_claim_self_contained_requests": 0,
    "prose_claim_wrong_context_emissions": 0,
    "protected_control_split_boundaries": 0,
    "source_message_ids_seen": [
      9271604311,
      9271604312
    ],
    "source_record_parse_failures": 0,
    "table_claim_exact_context_emissions": 1,
    "table_claim_exact_context_requests": 2,
    "table_claim_requests": 2,
    "table_claim_self_contained_requests": 0,
    "table_claim_wrong_context_emissions": 0
  },
  "normal_pass_completion_calls": 8,
  "prose_boundary_claim_sha256": "aeebb66034c889372c5f7a81cabd8891fc0126f9ad286825663ac3a047c4d17a",
  "required_before_indexing": true,
  "scope": "once_per_pending_run_configuration",
  "source_content_chars": 10349,
  "source_message_ids": [
    9271604311,
    9271604312
  ],
  "source_split_policy_version": "hymem-source-semantic-split-v10",
  "store_writes": 0,
  "structural_control_probe_version": "hymem-phase1-markdown-atom-split-probe-v1",
  "table_continuation_claim_sha256": "b0101bc38d69ab81541575a131a2d202037f0b8f489b9e07bfa22baf943cc13d",
  "usage_accounting": "excluded_from_scored_usage_dedicated_memory_pipeline_client",
  "version": "hymem-phase1-extraction-canary-v17"
}''')
_V17 = "hymem-phase1-extraction-canary-v17"
_V18 = "hymem-phase1-extraction-canary-v18"
_V19 = "hymem-phase1-extraction-canary-v19"
_V20 = "hymem-phase1-extraction-canary-v20"
_SCHEMA = "hymem-extraction-contract-sha256-v1"


def validate_archived_canary_config(policy: object, effective_config: object) -> dict:
    """Validate recorded commitments and their cross-links, not their preimages."""
    if not isinstance(policy, Mapping) or not isinstance(effective_config, Mapping):
        raise _integrity("archived policy/config is absent")
    version = policy.get("version")
    if version not in {_V17, _V18, _V19, _V20}:
        raise _integrity("archived policy version is unsupported")
    prompt = policy.get("prompt_version")
    binding = policy.get("extraction_contract")
    if (type(prompt) is not str or not prompt or prompt != prompt.strip()
            or len(prompt) > 128 or any(ord(char) < 32 or ord(char) == 127 for char in prompt)
            or not isinstance(binding, Mapping)
            or set(binding) != {"schema", "prompt_version", "identity"}
            or binding.get("schema") != _SCHEMA
            or binding.get("prompt_version") != prompt
            or type(binding.get("identity")) is not str
            or re.fullmatch(_SCHEMA + r":[0-9a-f]{64}", binding["identity"]) is None):
        raise _integrity("archived extraction commitment is malformed")
    if (effective_config.get("prompt_version") != prompt
            or not _strict_equal(effective_config.get("extraction_contract"), dict(binding))):
        raise _integrity("archived extraction commitment cross-link differs")
    expected = deepcopy(_V17_POLICY)
    if version in {_V18, _V19, _V20}:
        expected["version"] = version
        expected["normal_execution_path"]["provider_output_truncations"] = 0
    if version in {_V19, _V20}:
        # A new splitter policy is a new archive identity. The v17/v18 literal
        # commitments above remain byte-for-byte pinned to v10.
        expected["source_split_policy_version"] = "hymem-source-semantic-split-v11"
    if version == _V20:
        expected["contract_repair_policy_version"] = "hymem-source-only-contract-repair-v1"
    expected.update(prompt_version=prompt, extraction_contract=dict(binding))
    if not _strict_equal(dict(policy), expected):
        raise _integrity("archived policy differs from known version")
    return expected


def validate_archived_canary_report(value: object, *, policy: object,
                                   effective_config: object, **context) -> dict:
    expected = validate_archived_canary_config(policy, effective_config)
    return _validate_report_with_policy(value, policy=expected, **context)
