import contextlib
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

spec=importlib.util.spec_from_file_location('triplets',Path(__file__).resolve().parents[1]/'scripts/verify_and_evaluate_triplets.py')
evaluation=importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluation)


class TripletTests(unittest.TestCase):
    def test_event_overlap_excludes_false_negative(self):
        rows=[{'ideology':'left'},{'ideology':'right'},{'ideology':'left'},{'ideology':'left'}]
        events=[frozenset(['a','b']),frozenset(['a']),frozenset(['b']),frozenset(['c'])]
        pos,neg,stats=evaluation.build_triplet_pools(rows,events)
        np.testing.assert_array_equal(pos[0],[1]); np.testing.assert_array_equal(neg[0],[3])
        self.assertEqual(stats['potential_directed_triplets'],1)

    def test_distance_and_distinct_margin_conventions(self):
        z=np.array([[1.,0.],[.8,.6],[.675,np.sqrt(1-.675**2)]])
        v=evaluation.triplet_values(z,np.array([[0,1,2]]),.2)
        self.assertAlmostEqual(v['d_pos'][0],.4)
        self.assertAlmostEqual(v['d_neg'][0],.65)
        self.assertGreater(v['margin_gap'][0],0)
        self.assertLess(v['training_margin_gap'][0],0)

    def test_sample_determinism_and_no_repeated_articles(self):
        rows=[{'ideology':side} for e in range(20) for side in ('left','right')]
        events=[frozenset([str(e)]) for e in range(20) for _ in range(2)]
        pos,neg,_=evaluation.build_triplet_pools(rows,events)
        a=evaluation.sample_triplets(pos,neg,5,42); b=evaluation.sample_triplets(pos,neg,5,42)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(set(a.flatten())),15)
        for t in a: evaluation.verify_triplet(t,rows,events)

    def test_constant_paired_change_ci(self):
        draws=np.random.default_rng(43).integers(10,size=(100,10))
        self.assertEqual(evaluation.bootstrap_mean_ci(np.full(10,.5),draws),[.5,.5])

    def test_saved_triplet_reuse_and_tampering(self):
        rows=[{'sample_id':str(i),'ideology':('left' if i%2==0 else 'right')} for i in range(40)]
        events=[frozenset([str(i//2)]) for i in range(40)]
        pos,neg,_=evaluation.build_triplet_pools(rows,events)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=SimpleNamespace(results_dir=root,num_triplets=5,seed=42,resample_triplets=False)
            a,reused=evaluation.fixed_triplets(args,rows,events,pos,neg,'metadata-hash')
            self.assertFalse(reused)
            evaluation.write_json(root/'triplet_eval_manifest.json',{
                'metadata_sha256':'metadata-hash','seed':42,'triplet_count':5,
                'sampled_triplets_sha256':evaluation.digest(root/'sampled_triplets.csv')})
            b,reused=evaluation.fixed_triplets(args,rows,events,pos,neg,'metadata-hash')
            self.assertTrue(reused); np.testing.assert_array_equal(a,b)
            with (root/'sampled_triplets.csv').open('a') as f: f.write('\n')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                evaluation.fixed_triplets(args,rows,events,pos,neg,'metadata-hash')

    def test_metadata_permutation_and_nonfinite_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[]
            for i,side in enumerate(('left','right')):
                body=f'Original test article {i}'
                rows.append({'sample_id':f'{side}_000000','ideology':side,'original_row_index':0,
                    'source_file':f'{side}_neutral_dedup.jsonl','original_article':body,
                    'original_article_sha256':hashlib.sha256(body.encode()).hexdigest(),'event_ids':['event']})
            manifest={'samples':2,'left_samples':1,'right_samples':1,'dtype':'float32','hidden_dim':2,
                'shapes':{'lambda_0':[2,2]},'checkpoints':{'lambda_0':'test/checkpoint'},
                'original_article_fields':{s:f'{s}_original.body' for s in ('left','right')},
                'ignored_rewrite_fields':{s:f'{s}_neutralised' for s in ('left','right')}}
            evaluation.write_json(root/'manifest.json',manifest)
            pq.write_table(pa.Table.from_pylist(rows[::-1]),root/'metadata.parquet')
            with self.assertRaisesRegex(ValueError,'ordering contradicts'):
                evaluation.audit_inputs(root,{'warnings':[]})
            pq.write_table(pa.Table.from_pylist(rows),root/'metadata.parquet')
            np.save(root/'lambda_0_embeddings.npy',np.array([[1.,0.],[np.nan,1.]],dtype=np.float32))
            with self.assertRaisesRegex(ValueError,'Nonfinite embeddings'):
                evaluation.audit_inputs(root,{'warnings':[]})

    def test_failed_audit_does_not_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with contextlib.redirect_stderr(io.StringIO()):
                status=evaluation.main(['--data-dir',str(root),'--results-dir',str(root/'results')])
            self.assertEqual(status,1)
            self.assertTrue((root/'results/embedding_audit.json').exists())
            self.assertFalse((root/'results/triplet_separation.csv').exists())


if __name__=='__main__': unittest.main()
