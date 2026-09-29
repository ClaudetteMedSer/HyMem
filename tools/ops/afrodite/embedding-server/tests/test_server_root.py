"""Independent parent controls of batching, failure atomicity and locking."""
import importlib.util
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('root_server', Path(__file__).parents[1]/'server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class RootControls(unittest.TestCase):
    def test_original_failure_shape_bounded_without_real_inference(self):
        calls=[]
        class Model:
            def embed(self, texts, *, batch_size):
                calls.append((len(texts),batch_size))
                return ([float(x)]+[0.]*(server.DIM-1) for x in texts)
        with patch.object(server,'embed_model',Model()):
            result=server.embeddings(server.EmbeddingRequest(input=[str(i) for i in range(1125)]))
        self.assertEqual(len(calls),71)
        self.assertTrue(all(0<n<=16 and batch==16 for n,batch in calls))
        self.assertEqual([v.index for v in result.data],list(range(1125)))
        self.assertEqual([v.embedding[0] for v in result.data],list(range(1125)))

    def test_later_batch_error_is_atomic_and_releases_lock(self):
        class Model:
            calls=0
            def embed(self,texts,*,batch_size):
                self.calls+=1
                if self.calls==2: raise RuntimeError('PRIVATE_SYNTHETIC_SENTINEL')
                return ([0.]*server.DIM for _ in texts)
        model=Model()
        with patch.object(server,'embed_model',model):
            with self.assertRaises(server.HTTPException) as e:
                server.embeddings(server.EmbeddingRequest(input=['x']*17))
            self.assertEqual((e.exception.status_code,e.exception.detail),(503,'Embedding inference failed'))
            outcomes=[]
            t=threading.Thread(target=lambda:outcomes.append(server.embeddings(server.EmbeddingRequest(input='y'))))
            t.start();t.join(2)
            self.assertFalse(t.is_alive());self.assertEqual(len(outcomes),1)

    def test_two_embedding_requests_do_not_infer_in_parallel(self):
        entered=threading.Event(); release=threading.Event(); calls=[]; results=[]
        class Model:
            def embed(self,texts,*,batch_size):
                calls.append(texts[0]);entered.set()
                if texts[0]=='first':
                    if not release.wait(2): raise RuntimeError('timeout')
                return ([0.]*server.DIM for _ in texts)
        def go(label):results.append(server.embeddings(server.EmbeddingRequest(input=label)))
        with patch.object(server,'embed_model',Model()):
            a=threading.Thread(target=go,args=('first',));b=threading.Thread(target=go,args=('second',))
            a.start();self.assertTrue(entered.wait(1));b.start()
            try:
                time.sleep(.05);self.assertEqual(calls,['first'])
            finally:release.set()
            a.join(2);b.join(2)
        self.assertEqual(len(results),2);self.assertEqual(calls,['first','second'])

    def test_rejected_rerank_never_loads_model(self):
        with patch.object(server,'_load_reranker',side_effect=AssertionError('unexpected load')):
            for docs in ([], ['x']*(server.RERANK_MAX_DOCS+1)):
                with self.assertRaises(server.HTTPException) as e:
                    server.rerank(server.RerankRequest(query='q',documents=docs))
                self.assertEqual(e.exception.status_code,400)


if __name__=='__main__':unittest.main()
