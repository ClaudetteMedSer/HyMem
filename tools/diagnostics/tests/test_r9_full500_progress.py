import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).parents[1] / 'lme_r9_full500_progress.py'
SPEC = importlib.util.spec_from_file_location('full500_progress', PATH)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)

class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'question_ids': ['q' + str(n) for n in range(500)],
                         'schema': 'stock-lme-r9-full500-candidate-regression-v1',
                         'question_count': 500, 'sample': 0, 'source_indices': list(range(500))}
        self.cp = dict(expected_ids=self.manifest['question_ids'], scored=True,
                       verdict_key='correct', status='running', entries={})

    def test_absent_is_unknown_not_zero_usage(self):
        result = p.project(self.manifest, None)
        self.assertIsNone(result['inflight_usage'])
        self.assertNotIn('per_role_usage', result)

    def test_closed_projection(self):
        self.cp['entries']['q0'] = {'status': 'completed', 'row': {
            'question_id': 'q0', 'correct': True, 'answer': 'PRIVATE ANSWER',
            'indexing': {'complete': True, 'healthy': True, 'final_status': {
                'summary_health': {'summary_degraded_sessions': 2}}}}}
        self.cp['entries']['q1'] = {'status': 'failed', 'row': {'question_id': 'q1', 'correct': None}}
        self.cp['execution_segments'] = [{'elapsed_s': 12, 'reader_usage': {
            'calls': 3, 'calls_available': True, 'secret': 'PRIVATE SECRET'}}]
        result = p.project(self.manifest, self.cp)
        self.assertEqual(result['counts']['completed'], 1)
        self.assertEqual(result['counts']['failed'], 1)
        self.assertEqual(result['counts']['correct'], 1)
        self.assertEqual(result['counts']['healthy_indexing'], 1)
        self.assertEqual(result['counts']['summary_degraded_sessions'], 2)
        self.assertEqual(result['usage_scope'], 'checkpoint_snapshot')
        self.assertNotIn('PRIVATE', repr(result))

    def test_rejects_wrong_sample_and_ids(self):
        self.manifest['sample'] = 8
        with self.assertRaises(ValueError): p.project(self.manifest, self.cp)
        with self.assertRaises(ValueError): p.project({'question_ids': ['q0'] * 500}, None)

    def test_rejects_incomplete_terminal(self):
        self.cp['status'] = 'complete'
        with self.assertRaises(ValueError): p.project(self.manifest, self.cp)

    def test_rejects_nonfinite_usage(self):
        with self.assertRaises(ValueError): p.usage({'cost_usd': float('nan')})

    def test_canary_status_and_failure_reason_closed(self):
        self.cp['execution_segments'] = [{'elapsed_s': 0, 'extraction_canary': {
            'status': 'failed', 'failure_reason': 'call_failure', 'failure_details': ['PRIVATE']}}]
        result = p.project(self.manifest, self.cp)
        self.assertEqual(result['canary_status'], 'failed')
        self.assertEqual(result['canary_failure_reason'], 'call_failure')
        self.assertNotIn('PRIVATE', repr(result))
        self.cp['execution_segments'][0]['extraction_canary']['status'] = 'PRIVATE'
        with self.assertRaises(ValueError): p.project(self.manifest, self.cp)

    def test_continuation_failure_is_visible_without_raw_error(self):
        result = p.continuation_projection({'status': 'operator_pending',
            'paid_runs_started': None, 'validation_runs_started': 0,
            'operator_inspection_required': True, 'error_type': 'PRIVATE'})
        self.assertTrue(result['operator_inspection_required'])
        self.assertIsNone(result['paid_runs_started'])
        self.assertNotIn('PRIVATE', repr(result))
        with self.assertRaises(ValueError): p.continuation_projection({'status': 'unknown'})

    def test_terminal_projection_is_closed_and_bound(self):
        data = {'manifest_sha256': 'pin', 'question_ids': self.manifest['question_ids'],
                'status': 'scored_full500_completed', 'elapsed_s': 123,
                'counts': dict(expected=500, attempted=500, completed=500, failed=0,
                               missing=0, unique_attempted=500, total_attempts=500),
                'aggregate_paid_usage': {'calls': 501}, 'per_role_usage': {'retrieval': {'calls': 0}},
                'per_question': [{'question_id': q, 'answer_correct': True, 'answer': 'PRIVATE'}
                                 for q in self.manifest['question_ids']]}
        for key in ('strict_scored_artifact_validated', 'physical_checkpoint_bound',
                    'process_completed_cleanly', 'benchmark_completed_without_faults',
                    'reader_judge_calls_measured', 'canary_accounted_separately'):
            data[key] = True
        result = p.terminal_projection(data, self.manifest, 'pin')
        self.assertEqual(result['correct'], 500)
        self.assertIn('retrieval', result['per_role_usage'])
        self.assertNotIn('PRIVATE', repr(result))
        with self.assertRaises(ValueError): p.terminal_projection(data, self.manifest, 'wrong-pin')

    def test_finalizer_process_identity_without_hiding_continuation(self):
        owned = {'pid': 123, 'process_group_id': 123, 'session_id': 123, 'start_ticks': 999}
        fields = ['S', '1', '123', '123'] + ['0'] * 15 + ['999']
        stat = '123 (private process name) ' + ' '.join(fields)
        result = p.process_projection(owned, stat)
        self.assertTrue(result['identity_verified'])
        self.assertTrue(result['process_live'])
        self.assertFalse(p.process_projection(owned, None)['process_present'])
        owned['start_ticks'] = 998
        with self.assertRaises(ValueError): p.process_projection(owned, stat)
        original = p.continuation_projection({'status': 'operator_pending',
                                              'operator_inspection_required': True})
        completed = p.continuation_projection({'status': 'offline_validation_finished',
                                               'validation_exit_code': 0})
        self.assertTrue(original['operator_inspection_required'])
        self.assertEqual(completed['validation_exit_code'], 0)

if __name__ == '__main__':
    unittest.main()
