import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import unittest

spec=importlib.util.spec_from_file_location('r9_core',Path(__file__).with_name('core.py'))
core=importlib.util.module_from_spec(spec); spec.loader.exec_module(core)

class CoreTests(unittest.TestCase):
    def test_capture_escapes_exception_handlers(self):
        request=object()
        def extract(conn,sid,client,**kw):
            try: client.complete(request)
            except Exception: self.fail('caught by product Exception handler')
        self.assertIs(core.capture_normal(None,'sid',object(),extract),request)
    def test_exact_source_not_placeholder(self):
        primary=SimpleNamespace(user='{"source":"private hostile \\""}',max_tokens=3072,temperature=0,response_format='json')
        def builder(request,feedback):
            self.assertEqual(feedback,'X'*503)
            return SimpleNamespace(user=json.dumps({'original_generation_input':request.user}),max_tokens=3072,temperature=0,response_format='json')
        self.assertEqual(json.loads(core.source_exact_repair(primary,builder).user)['original_generation_input'],primary.user)
    def test_budget_reserves_http_retries_and_phase_single_use(self):
        class Client:
            request_attempts=0
            def complete(self,request): self.request_attempts+=3; return 'ok'
        budget=core.Budget(Client(),SimpleNamespace(check=lambda:None,remaining=lambda:900),'hash',SimpleNamespace())
        budget.start('normal')
        self.assertEqual(budget.complete(None),'ok'); budget.complete(None)
        with self.assertRaisesRegex(RuntimeError,'completion_budget'): budget.complete(None)
        with self.assertRaisesRegex(RuntimeError,'phase_single_use'): budget.start('normal')
        self.assertEqual(budget.calls,2)
        budget.start('source_exact_repair'); budget.client.request_attempts=34
        with self.assertRaisesRegex(RuntimeError,'completion_budget'): budget.complete(None)
        self.assertEqual(budget.calls,2)
    def test_projection_private_text_absent(self):
        raw='{"alternatives":["secret1","secret2","secret3"]}'
        public=core.parse_projection(raw,lambda _:('secret1',None))
        self.assertNotIn('secret',json.dumps(public))
        self.assertEqual(public['option_lengths'],[7,7,7])
        invented=core.parse_projection(raw,lambda _:('secret1',None),invented=True)
        self.assertEqual(invented['selected'],'secret1')
    def test_backup_rejects_reuse(self):
        import tempfile,sqlite3
        with tempfile.TemporaryDirectory() as directory:
            original=Path(directory)/'original.sqlite'; clone=Path(directory)/'clone.sqlite'
            conn=sqlite3.connect(original)
            conn.execute('CREATE TABLE x(v)'); conn.commit(); conn.close()
            core.backup(original,clone)
            with self.assertRaisesRegex(RuntimeError,'fresh_clone_required'): core.backup(original,clone)
    def test_selection_requires_exact_source_state(self):
        import hashlib
        sid='invented'
        row=dict(id=sid,auto_summary=None,summary_failure_reason='summary_output_cap',summary_failure_count=1,
                 digest_published_message_id=300,coverage_message_id=300)
        messages=[dict(id=i,content='x'*(6996 if i==293 else 0)) for i in range(293,301)]
        class Cursor:
            def __init__(self,value): self.value=value
            def fetchall(self): return self.value
            def fetchone(self): return self.value
        class Conn:
            def execute(self,sql,args=()):
                return Cursor([row] if 'SELECT * FROM sessions' in sql else messages if 'FROM messages' in sql else None)
        old=core.TARGET
        core.TARGET=hashlib.sha256(sid.encode()).hexdigest()
        try:
            self.assertEqual(core.select(Conn()),sid)
            row['summary_failure_count']=2
            with self.assertRaisesRegex(RuntimeError,'target_state'): core.select(Conn())
            row['summary_failure_count']=1; messages[0]['content']='x'
            with self.assertRaisesRegex(RuntimeError,'target_source'): core.select(Conn())
        finally: core.TARGET=old

    def test_expired_phase_cannot_dispatch(self):
        client=SimpleNamespace(request_attempts=0,complete=lambda _:self.fail('paid dispatch'))
        budget=core.Budget(client,SimpleNamespace(check=lambda:None,remaining=lambda:900),'hash',SimpleNamespace())
        budget.start('normal'); budget.phase_end=0
        with self.assertRaisesRegex(RuntimeError,'invocation_deadline'): budget.complete(None)
        self.assertEqual(budget.calls,0)
    def test_native_deadline_is_bounded_to_phase(self):
        from hymem.deadline import current_deadline
        seen=[]
        def complete(_):
            seen.append(current_deadline().remaining())
            return 'ok'
        client=SimpleNamespace(request_attempts=0,complete=complete)
        budget=core.Budget(client,SimpleNamespace(check=lambda:None,remaining=lambda:900),'hash',SimpleNamespace())
        budget.start('normal'); budget.complete(None)
        self.assertTrue(0<seen[0]<=120)
    def test_unknown_parse_labels_are_closed(self):
        result=core.parse_projection(object(),lambda _: (None,'private arbitrary response'))
        self.assertEqual(result['failure_reason'],'unrecognized_failure')
        self.assertEqual(result['raw_type'],'other')
        self.assertNotIn('private',json.dumps(result))
        for label in ('parse_failure','shape_failure','summary_shape_failure'):
            self.assertEqual(core.parse_projection(None,lambda _: (None,label))['failure_reason'],label)
    def test_backup_closes_source_when_target_connect_fails(self):
        import tempfile
        from unittest.mock import patch
        source=SimpleNamespace(close=unittest.mock.Mock(),rollback=unittest.mock.Mock())
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(core.sqlite3,'connect',side_effect=[source,RuntimeError('target')]):
                with self.assertRaises(RuntimeError): core.backup(Path(directory)/'source',Path(directory)/'clone')
        source.close.assert_called_once()
    def test_failure_evidence_does_not_export_exception_message(self):
        try:
            raise RuntimeError('PRIVATE RESPONSE SOURCE CREDENTIAL')
        except RuntimeError as exc:
            evidence=core.failure_evidence(exc)
        self.assertEqual(evidence['exception_class'],'RuntimeError')
        self.assertNotIn('PRIVATE',json.dumps(evidence))
        self.assertTrue(all(set(frame)=={'file','line'} for frame in evidence['frames']))
    def test_pinned_wal_snapshot_avoids_restart_loop(self):
        import sqlite3,tempfile
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source'; dest=Path(directory)/'old'
            writer=sqlite3.connect(source,isolation_level=None)
            writer.execute('PRAGMA journal_mode=WAL');writer.execute('CREATE TABLE x(v)')
            writer.executemany('INSERT INTO x VALUES(?)',[('x'*4096,) for _ in range(12)])
            def run(pinned):
                reader=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,isolation_level=None)
                target=sqlite3.connect(dest if not pinned else Path(directory)/'pinned')
                calls=[0]
                try:
                    reader.execute('PRAGMA query_only=ON')
                    if pinned: reader.execute('BEGIN');reader.execute('SELECT count(*) FROM sqlite_schema').fetchone()
                    def callback(*_):
                        calls[0]+=1
                        writer.execute('UPDATE x SET v=? WHERE rowid=1',(str(calls[0])*4096,))
                        if calls[0]>50:raise RuntimeError('restart_loop')
                    reader.backup(target,pages=1,progress=callback)
                finally: target.close();reader.close()
                return calls[0]
            try:
                with self.assertRaisesRegex(RuntimeError,'restart_loop'): run(False)
                self.assertLess(run(True),50)
            finally: writer.close()

if __name__=='__main__': unittest.main()
