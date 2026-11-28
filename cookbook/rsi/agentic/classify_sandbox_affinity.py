# Copyright (c) ModelScope Contributors. All rights reserved.
"""Build a deterministic, advisory sandbox-affinity overlay for a frozen bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cookbook.rsi.agentic.profiles import ProfileRegistry, default_profile_registry
from cookbook.rsi.agentic.task_contract import TaskIntent, resolve_intent

EXPECTED_BANK_SHA256 = '91c6a69dbdd23d031bf94e885055f2e6e0e47813c6949a400bcbab66c93f6893'
CLASSIFIER_VERSION = 'sandbox-affinity-rules-v2'
EXPECTED_BANK_COUNT = 429
POLICY = ('Affinities are advisory task-generation directions inferred only from frozen topic titles. '
          'They never grant tools or capabilities. Every card permits objective knowledge questions; '
          'other profiles require a later TaskIntent based on concrete actions, inputs, outputs, and verifier. '
          'Direct physical work must be reframed as knowledge, simulation, or supplied-data analysis.')
_COMMON_FIELDS = frozenset({
    'card_id',
    'checkpoint_sha256',
    'created_unix',
    'decode_params',
    'finish_reason',
    'generated_token_count',
    'global_index',
    'knowledge',
    'model',
    'prompt_sha256',
    'schema_version',
    'seed',
    'token_count',
    'tokenizer_sha256',
    'topic_path',
})
_SHARDED_FIELDS = frozenset({'local_index', 'num_shards', 'shard_id'})
_MINED_FIELDS = frozenset({'attempt', 'domain', 'epistemic_status', 'knowledge_sha256', 'phase', 'title'})
_CARD_SCHEMAS = (_COMMON_FIELDS | _SHARDED_FIELDS, _COMMON_FIELDS | _MINED_FIELDS)
_REQUIRED_FIELDS = _CARD_SCHEMAS[0]
_CARD_ID = re.compile(r'^[0-9a-f]{24}$')
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_PROFILE_ORDER = (
    'knowledge-answer@1',
    'python-compute@1',
    'lean-proof@1',
    'code-python@1',
    'sql-analytics@1',
    'db-analyst@1',
    'data-wrangling@1',
    'docs-retrieval@1',
    'web-research@1',
    'log-forensics@1',
    'repo-patch@1',
    'world-forensics@1',
)
_PROFILE_FORMS = {
    'knowledge-answer@1': ('knowledge_answer', ),
    'python-compute@1': ('given_data_analysis', 'numeric_problem', 'symbolic_problem'),
    'lean-proof@1': ('formal_proof', ),
    'code-python@1': ('python_implementation', ),
    'sql-analytics@1': ('database_analysis', ),
    'db-analyst@1': ('database_analysis', 'grant_aware_query'),
    'data-wrangling@1': ('data_cleaning', ),
    'docs-retrieval@1': ('retrieval_qa', ),
    'web-research@1': ('web_research', 'source_verification'),
    'log-forensics@1': ('incident_analysis', ),
    'repo-patch@1': ('bug_fix', ),
    'world-forensics@1': ('incident_analysis', 'cross_source_reconciliation'),
}
_RULES: Dict[str, Tuple[re.Pattern[str], ...]] = {
    'numeric':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\bcalculat\w*\b',
            r'\bquantitative\b',
            r'\bestimat\w*\b',
            r'\bprobabilit\w*\b',
            r'\brate(?:s)?\b',
            r'\bdensit\w*\b',
            r'\bthreshold\w*\b',
            r'\bmetric\w*\b',
            r'\bequations?\b',
            r'\boptimization\b',
            r'\boptimal\b',
            r'\bnumerical\b',
            r'\btrajectory\b',
            r'\bcurvature\b',
            r'\bentropy\b',
            r'\btemperature\b',
            r'\bpressure\b',
            r'\bvelocity\b',
            r'\bconcentration\b',
            r'\bisotop\w*\b',
            r'\bconductivity\b',
        )),
    'data':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\banalys\w*\b',
            r'\bmeasure(?:ment|ments)?\b',
            r'\btime[- ]series\b',
            r'\bcorrelation\w*\b',
            r'\bdistribution\w*\b',
            r'\bpattern\w*\b',
            r'\bspectral?\b',
            r'\bmap(?:ping|s)?\b',
            r'\bsequence(?:s|ing)?\b',
            r'\bmodel(?:ing|ling)?\b',
            r'\bsimulation\w*\b',
        )),
    'symbolic':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\bgeometry\b',
            r'\btheorem\b',
            r'\bmanifold\w*\b',
            r'\boperator\b',
            r'\blie algebra\w*\b',
            r'\bgeodesic\w*\b',
            r'\bgroup\b',
            r'\bfock space\b',
            r'\bvector fields?\b',
            r'\bnon[- ]euclidean\b',
            r'\bquantization\b',
        )),
    'formal':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\btheorem\b',
            r'\bmanifold\w*\b',
            r'\blie algebra\w*\b',
            r'\bgeodesic equations?\b',
            r'\bspectral triples?\b',
            r'\bpseudogroup\w*\b',
            r'\boperator ordering\b',
            r'\bisometric embedding\b',
            r'\bconformal mapping\b',
        )),
    'code':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\balgorithm\w*\b',
            r'\bimplementation\b',
            r'\bpipeline\w*\b',
            r'\bdiscretization\b',
            r'\bnumerical schemes?\b',
            r'\bpath planning\b',
            r'\bstate estimation\b',
            r'\bsensor fusion\b',
            r'\bcollision avoidance\b',
            r'\bfinite[- ]time stabilization\b',
            r'\bcontrol theory\b',
        )),
    'web':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\bcase law\b',
            r'\blegal frameworks?\b',
            r'\bstatutes?\b',
            r'\bconstitutional amendments?\b',
            r'\bsupreme court\b',
            r'\bjudicial conduct\b',
            r'\blifetime tenure\b',
            r'\bweb[- ]based\b',
            r'\bweb search\b',
            r'\bweb archive\b',
            r'\bweb sources?\b',
            r'\bweb pages?\b',
            r'\bweb data\b',
            r'\bfrom web\b',
            r'\bacross web\b',
            r'\bpublic web\b',
            r'\bsearch engines?\b',
            r'\bsearch results?\b',
            r'\bonline sources\b',
            r'\binformation retrieval\b',
            r'\bfact[- ]checking\b',
            r'\bclaim verification\b',
            r'\bsource credibility\b',
            r'\binformation authority\b',
            r'\binformation provenance\b',
            r'\bacademic resource discovery\b',
            r'\bcompetitive intelligence\b',
        )),
    'sql':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\bsql\b',
            r'\bdatabases?\b',
            r'\bstored procedures?\b',
            r'\bcommon table expressions?\b',
            r'\bwindow functions?\b',
            r'\bjoin operations?\b',
            r'\bsubquer\w+\b',
            r'\brecursive query\b',
            r'\bpartitioning and sharding\b',
            r'\breferential integrity\b',
            r'\bquery optimization\b',
            r'\bschema design\b',
            r'\bschema evolution\b',
        )),
    'wrangling':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\bdata cleaning\b',
            r'\bvalidation techniques\b',
            r'\bdata quality\b',
            r'\bdeduplication\b',
            r'\bmissing values\b',
            r'\bmalformed\b',
        )),
    'incident':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\bincidents?\b',
            r'\blog analysis\b',
            r'\bforensics?\b',
            r'\broot cause\b',
            r'\baudit trail\b',
            r'\bpost-?mortem\b',
        )),
    'bugfix':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\bbug fixes?\b',
            r'\bfailing tests?\b',
            r'\bunit tests?\b',
            r'\brefactor\w*\b',
            r'\bcode review\b',
            r'\bdefect tracking\b',
        )),
    'lab_measurement':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\bmeasure(?:ment|ments)?\b',
            r'\bspectroscop\w*\b',
            r'\bspectrograph\w*\b',
            r'\bdiffraction\b',
            r'\bsensor\w*\b',
            r'\bdetector\w*\b',
            r'\bsamples?\b',
            r'\bsequencing\b',
            r'\bimaging\b',
            r'\bcalibration\b',
            r'\btomography\b',
            r'\bpulsed microwave\b',
        )),
    'field_site_work':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\barchaeolog\w*\b',
            r'\bsediment cores?\b',
            r'\bethnograph\w*\b',
            r'\bsite stabilization\b',
            r'\brestoration\b',
            r'\bsalvage\b',
            r'\bdecommissioning\b',
            r'\bdeployment\b',
        )),
    'hardware_operation':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\bqubits?\b',
            r'\bcircuits?\b',
            r'\bMRI\b',
            r'\bcoils?\b',
            r'\bSQUID\b',
            r'\bAUVs?\b',
            r'\bautonomous underwater\b',
            r'\bsonar\b',
            r'\bdetector\w*\b',
            r'\bsensor\w*\b',
        )),
    'human_subject':
    tuple(
        re.compile(pattern, re.IGNORECASE) for pattern in (
            r'\bergonomic\w*\b',
            r'\btyping sessions?\b',
            r'\bposture\b',
            r'\bchildren\b',
            r'\bspeech communit\w*\b',
            r'\bethnograph\w*\b',
            r'\bcommunity[- ]led\b',
        )),
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _matched_rules(text: str, family: str) -> List[str]:
    return [pattern.pattern for pattern in _RULES[family] if pattern.search(text)]


_DOMAIN_FAMILY = {'sql': 'sql', 'web': 'web'}


def read_frozen_cards(bank_path: Path,
                      expected_sha256: str = EXPECTED_BANK_SHA256,
                      expected_count: int = EXPECTED_BANK_COUNT) -> List[Tuple[dict, str]]:
    before = _sha256_file(bank_path)
    if before != expected_sha256:
        raise ValueError(f'frozen bank SHA-256 mismatch: expected {expected_sha256}, got {before}')
    records: List[Tuple[dict, str]] = []
    with bank_path.open(encoding='utf-8') as handle:
        for line_number, raw_line in enumerate(handle, 1):
            serialized = raw_line.rstrip('\r\n')
            if not serialized:
                raise ValueError(f'blank record at line {line_number}')
            card = json.loads(serialized)
            fields = set(card)
            if not any(fields == schema for schema in _CARD_SCHEMAS):
                nearest = min(_CARD_SCHEMAS, key=lambda schema: len(schema ^ fields))
                raise ValueError(f'invalid card fields at line {line_number}: matches neither the '
                                 f'sharded nor the incrementally-mined layout; '
                                 f'missing={sorted(nearest - fields)}, extra={sorted(fields - nearest)}')
            if card['schema_version'] != 1:
                raise ValueError(f'card {card.get("card_id")!r} is not schema version 1')
            if not _CARD_ID.fullmatch(card['card_id']):
                raise ValueError(f'card id at line {line_number} must be 24 lowercase hexadecimal characters')
            for field in ('checkpoint_sha256', 'prompt_sha256', 'tokenizer_sha256'):
                if not _SHA256.fullmatch(card[field]):
                    raise ValueError(f'card {card["card_id"]!r} has invalid {field}')
            if not isinstance(card['global_index'], int) or isinstance(card['global_index'], bool):
                raise ValueError(f'card {card["card_id"]!r} has invalid global_index')
            topic = card['topic_path']
            if not isinstance(topic, dict) or set(topic) != {'root', 'leaf'}:
                raise ValueError(f'card {card.get("card_id")!r} has invalid topic_path')
            if not all(isinstance(topic[key], str) and topic[key].strip() for key in ('root', 'leaf')):
                raise ValueError(f'card {card.get("card_id")!r} has an empty topic title')
            records.append((card, _sha256_bytes(serialized.encode('utf-8'))))
    if len(records) != expected_count:
        raise ValueError(f'expected {expected_count} cards, found {len(records)}')
    ids = [card['card_id'] for card, _ in records]
    indices = [card['global_index'] for card, _ in records]
    if len(ids) != len(set(ids)):
        raise ValueError('card ids must be unique')
    if len(indices) != len(set(indices)):
        raise ValueError('global indices must be unique')
    if sorted(indices) != list(range(expected_count)):
        raise ValueError('global indices must be contiguous from zero')
    after = _sha256_file(bank_path)
    if after != before:
        raise RuntimeError('frozen bank changed while it was being read')
    return sorted(records, key=lambda item: item[0]['global_index'])


def classify_card(card: dict, source_record_sha256: str, registry: ProfileRegistry) -> dict:
    topic = card['topic_path']
    text = f'{topic["root"]}\n{topic["leaf"]}'
    matches = {family: _matched_rules(text, family) for family in _RULES}
    domain = str(card.get('domain') or '').strip().lower()
    if domain:
        if domain not in _DOMAIN_FAMILY:
            raise ValueError(f'card {card.get("card_id")!r} has domain {domain!r}, which no rule '
                             f'family routes; known domains: {sorted(_DOMAIN_FAMILY)}')
        family = _DOMAIN_FAMILY[domain]
        marker = f'domain:{domain}'
        if marker not in matches[family]:
            matches[family] = [marker] + list(matches[family])
    profiles = {'knowledge-answer@1'}
    reasons = ['baseline:objective_knowledge_question']
    confidence = 0.98

    compute_families = [family for family in ('numeric', 'data', 'symbolic') if matches[family]]
    if compute_families:
        profiles.add('python-compute@1')
        reasons.extend(f'{family}:{pattern}' for family in compute_families for pattern in matches[family])
        confidence = min(confidence, 0.82)
    if matches['formal']:
        profiles.add('lean-proof@1')
        reasons.extend(f'formal:{pattern}' for pattern in matches['formal'])
        confidence = min(confidence, 0.76)
    if matches['code']:
        profiles.add('code-python@1')
        reasons.extend(f'code:{pattern}' for pattern in matches['code'])
        confidence = min(confidence, 0.86)
    if matches['web']:
        profiles.add('web-research@1')
        profiles.add('docs-retrieval@1')
        reasons.extend(f'web:{pattern}' for pattern in matches['web'])
        confidence = min(confidence, 0.76)
    if matches['sql']:
        profiles.add('sql-analytics@1')
        profiles.add('db-analyst@1')
        reasons.extend(f'sql:{pattern}' for pattern in matches['sql'])
        confidence = min(confidence, 0.86)
    if matches['wrangling']:
        profiles.add('data-wrangling@1')
        reasons.extend(f'wrangling:{pattern}' for pattern in matches['wrangling'])
        confidence = min(confidence, 0.86)
    if matches['incident']:
        profiles.add('log-forensics@1')
        profiles.add('world-forensics@1')
        reasons.extend(f'incident:{pattern}' for pattern in matches['incident'])
        confidence = min(confidence, 0.80)
    if matches['bugfix']:
        profiles.add('repo-patch@1')
        reasons.extend(f'bugfix:{pattern}' for pattern in matches['bugfix'])
        confidence = min(confidence, 0.86)

    physical_constraints = []
    for family in ('lab_measurement', 'field_site_work', 'hardware_operation', 'human_subject'):
        if matches[family]:
            physical_constraints.append(family)
            reasons.extend(f'physical:{family}:{pattern}' for pattern in matches[family])
    if physical_constraints:
        physical_constraints.append('requires_reframing_to_given_inputs_or_simulation')
        confidence = min(confidence, 0.72)

    eligible_profiles = [profile_id for profile_id in _PROFILE_ORDER if profile_id in profiles]
    for profile_id in eligible_profiles:
        registry.get(profile_id)
    task_forms = {'knowledge_answer'}
    if matches['numeric']:
        task_forms.add('numeric_problem')
    if matches['data']:
        task_forms.add('given_data_analysis')
    if matches['symbolic']:
        task_forms.add('symbolic_problem')
    if matches['formal']:
        task_forms.add('formal_proof')
    if matches['code']:
        task_forms.add('python_implementation')
    if matches['web']:
        task_forms.add('web_research')
        task_forms.add('retrieval_qa')
        task_forms.add('source_verification')
    if matches['sql']:
        task_forms.add('database_analysis')
        task_forms.add('grant_aware_query')
    if matches['wrangling']:
        task_forms.add('data_cleaning')
    if matches['incident']:
        task_forms.add('incident_analysis')
        task_forms.add('cross_source_reconciliation')
    if matches['bugfix']:
        task_forms.add('bug_fix')
    return {
        'schema_version': 1,
        'card_id': card['card_id'],
        'global_index': card['global_index'],
        'source_record_sha256': source_record_sha256,
        'topic_path': {
            'root': topic['root'],
            'leaf': topic['leaf']
        },
        'eligible_profiles': eligible_profiles,
        'task_forms': sorted(task_forms),
        'physical_constraints': physical_constraints,
        'confidence': confidence,
        'reasons': sorted(set(reasons)),
        'advisory_only': True,
    }


def _routing_matrix(registry: ProfileRegistry) -> dict:
    cases = (
        ('objective_geography', 'knowledge-answer@1', 'knowledge_answer', ('objective_answer', ), ('given_text', ),
         ('answer', )),
        ('math_topic_objective_answer', 'knowledge-answer@1', 'knowledge_answer', ('objective_answer', ),
         ('given_text', ), ('answer', )),
        ('numeric_calculation', 'python-compute@1', 'numeric_problem', ('numeric_compute', ), ('given_text', ),
         ('answer', )),
        ('formal_theorem', 'lean-proof@1', 'formal_proof', ('formal_proof', ), ('given_text', ), ('proof', )),
        ('python_hidden_tests', 'code-python@1', 'python_implementation', ('python_code', 'workspace_files'),
         ('given_text', 'repository_fixture'), ('workspace', )),
        ('fresh_web_evidence', 'web-research@1', 'web_research', ('web_search', 'web_fetch'), ('given_text', ),
         ('answer', 'citations')),
        ('source_verification', 'web-research@1', 'source_verification', ('web_search', 'web_fetch', 'citation'),
         ('given_text', ), ('answer', 'citations')),
        ('sql_aggregate_analysis', 'sql-analytics@1', 'database_analysis', ('database_query', 'data_compute'),
         ('given_text', 'data_fixture'), ('answer', 'workspace')),
        ('grant_aware_db_query', 'db-analyst@1', 'grant_aware_query', ('database_query', 'database_schema'),
         ('given_text', ), ('answer', 'workspace')),
        ('data_cleaning_fixture', 'data-wrangling@1', 'data_cleaning', ('data_compute', 'text_processing'),
         ('given_text', 'data_fixture'), ('answer', 'workspace')),
        ('retrieval_qa_from_docs', 'docs-retrieval@1', 'retrieval_qa', ('document_retrieval', 'citation'),
         ('given_text', 'data_fixture'), ('answer', 'citations')),
        ('incident_log_forensics', 'log-forensics@1', 'incident_analysis', ('log_analysis', 'shell_forensics'),
         ('given_text', 'data_fixture'), ('answer', 'workspace')),
        ('repo_bug_fix', 'repo-patch@1', 'bug_fix', ('python_code', 'workspace_files', 'test_execution'),
         ('given_text', 'repository_fixture'), ('workspace', )),
        ('cross_source_reconciliation', 'world-forensics@1', 'cross_source_reconciliation',
         ('database_query', 'document_retrieval', 'log_analysis'), ('given_text', ), ('answer', 'workspace')),
        ('direct_physical_action', '', 'knowledge_answer', ('physical_action', ), ('given_text', ), ('answer', )),
    )
    results = []
    for case_id, requested, task_form, capabilities, inputs, outputs in cases:
        intent = TaskIntent(
            intent_id=f'gate-a-{case_id}',
            requested_profile_id=requested,
            task_form=task_form,
            capabilities=capabilities,
            input_kinds=inputs,
            output_kinds=outputs,
            reason='Gate A static routing acceptance case.',
        )
        try:
            resolution = resolve_intent(intent, registry)
        except ValueError as error:
            results.append({'case_id': case_id, 'status': 'rejected', 'error': str(error)})
        else:
            results.append({
                'case_id': case_id,
                'status': 'resolved',
                'resolved_profile_id': resolution.profile.profile_id,
                'profile_digest': resolution.profile.digest,
                'registry_digest': resolution.registry_digest,
                'router_version': resolution.router_version,
            })
    return {'schema_version': 1, 'cases': results}


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def _jsonl(records: Iterable[dict]) -> str:
    return ''.join(f'{_canonical_json(record)}\n' for record in records)


def generate_overlay(bank_path: Path,
                     output_dir: Path,
                     expected_sha256: str = EXPECTED_BANK_SHA256,
                     expected_count: int = EXPECTED_BANK_COUNT,
                     low_confidence_threshold: float = 0.8) -> dict:
    bank_path = bank_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir == bank_path.parent or bank_path.parent in output_dir.parents:
        raise ValueError('affinity output must not be written inside the frozen bank directory')
    registry = default_profile_registry()
    for profile_id, task_forms in _PROFILE_FORMS.items():
        profile = registry.get(profile_id)
        if not set(task_forms).issubset(profile.task_forms):
            raise RuntimeError(f'affinity task forms drifted from profile {profile_id!r}')
    records = read_frozen_cards(bank_path, expected_sha256, expected_count)
    overlays = [classify_card(card, record_sha, registry) for card, record_sha in records]
    ids = [record['card_id'] for record in overlays]
    if len(ids) != len(set(ids)):
        raise RuntimeError('classifier produced duplicate card ids')

    audits = []
    unavailable = {profile.profile_id for profile in registry.profiles() if not profile.available}
    for record in overlays:
        audit_reasons = []
        if record['confidence'] < low_confidence_threshold:
            audit_reasons.append('low_confidence')
        if record['physical_constraints']:
            audit_reasons.append('physical_constraint')
        if unavailable.intersection(record['eligible_profiles']):
            audit_reasons.append('unavailable_profile_candidate')
        if audit_reasons:
            audits.append({
                'schema_version': 1,
                'card_id': record['card_id'],
                'global_index': record['global_index'],
                'audit_reasons': audit_reasons,
                'eligible_profiles': record['eligible_profiles'],
                'physical_constraints': record['physical_constraints'],
                'confidence': record['confidence'],
            })

    routing = _routing_matrix(registry)
    overlay_text = _jsonl(overlays)
    audit_text = _jsonl(audits)
    routing_text = f'{json.dumps(routing, sort_keys=True, indent=2, ensure_ascii=False)}\n'
    artifacts = {
        'affinities.jsonl': overlay_text,
        'manual_audit.jsonl': audit_text,
        'routing_matrix.json': routing_text,
    }
    profile_counts = Counter(profile_id for record in overlays for profile_id in record['eligible_profiles'])
    task_form_counts = Counter(task_form for record in overlays for task_form in record['task_forms'])
    physical_counts = Counter(constraint for record in overlays for constraint in record['physical_constraints']
                              if constraint != 'requires_reframing_to_given_inputs_or_simulation')
    manifest = {
        'schema_version': 1,
        'status': 'ready',
        'source_bank': str(bank_path),
        'source_bank_sha256': expected_sha256,
        'source_bank_schema_version': 1,
        'card_count': len(overlays),
        'card_ids_sha256': _sha256_bytes(_canonical_json(ids).encode('utf-8')),
        'classifier_version': CLASSIFIER_VERSION,
        'classifier_source_sha256': _sha256_file(Path(__file__)),
        'policy_sha256': _sha256_bytes(POLICY.encode('utf-8')),
        'policy': POLICY,
        'registry_digest': registry.digest,
        'profile_digests': {
            profile.profile_id: profile.digest
            for profile in registry.profiles()
        },
        'profile_availability': {
            profile.profile_id: {
                'available': profile.available,
                'enabled_for_new_tasks': profile.enabled_for_new_tasks,
                'unavailable_reason': profile.unavailable_reason,
            }
            for profile in registry.profiles()
        },
        'artifacts': {
            name: _sha256_bytes(content.encode('utf-8'))
            for name, content in artifacts.items()
        },
        'summary': {
            'profile_counts': dict(sorted(profile_counts.items())),
            'task_form_counts': dict(sorted(task_form_counts.items())),
            'physical_constraint_counts': dict(sorted(physical_counts.items())),
            'manual_audit_count': len(audits),
            'low_confidence_count': sum(record['confidence'] < low_confidence_threshold for record in overlays),
            'physical_card_count': sum(bool(record['physical_constraints']) for record in overlays),
            'root_count': len({record['topic_path']['root']
                               for record in overlays}),
        },
    }
    manifest_text = f'{json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False)}\n'
    for name, content in artifacts.items():
        _write_atomic(output_dir / name, content)
    _write_atomic(output_dir / 'manifest.json', manifest_text)
    ready = {
        'schema_version': 1,
        'status': 'ready',
        'manifest_sha256': _sha256_bytes(manifest_text.encode('utf-8')),
    }
    _write_atomic(output_dir / 'READY', f'{_canonical_json(ready)}\n')
    if _sha256_file(bank_path) != expected_sha256:
        raise RuntimeError('frozen bank changed during overlay generation')
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-sha256', default=EXPECTED_BANK_SHA256)
    parser.add_argument('--expected-count', type=int, default=EXPECTED_BANK_COUNT)
    parser.add_argument('--low-confidence-threshold', type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 <= args.low_confidence_threshold <= 1:
        raise SystemExit('--low-confidence-threshold must be in [0, 1]')
    manifest = generate_overlay(args.bank, args.output_dir, args.expected_sha256, args.expected_count,
                                args.low_confidence_threshold)
    print(
        json.dumps(
            {
                'output_dir': str(args.output_dir),
                'source_bank_sha256': manifest['source_bank_sha256'],
                **manifest['summary'],
            },
            sort_keys=True,
            ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
