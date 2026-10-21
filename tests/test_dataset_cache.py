"""Dataset-cache integrity and fresh-runtime training checks; no real airfare data."""
import contextlib
import copy
from datetime import date
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dataset_cache as cache
from test_tabular_q import fixture, STATS

NOTEBOOK=Path(__file__).resolve().parents[1]/cache.NOTEBOOK_NAME
CELLS=json.loads(NOTEBOOK.read_text())['cells']


def namespace():
    module=ModuleType('dataset_cache_test_runtime')
    sys.modules[module.__name__]=module
    cache.bootstrap_definitions(module.__dict__,NOTEBOOK)
    return module.__dict__


def snapshot_fixture():
    _,bundle,data=fixture(trend=.02)
    ns=namespace()
    ns.update(KEY_COLUMNS=['legId','flightDate','searchDate'],TARGET_COL='fareChange',
              ANCHOR_COL='lagged_fare_1',LEVEL_COL='totalFare',DELTA_STD=.02,feature_dim=4,
              batch_size=4,SEQUENCE_LENGTH=10,N_DIFF_LAGS=20,MATCH_TRAINING_TARGETS=True,
              DL_TRAIN_WINDOW_STRIDE=7,DL_MAX_TRAIN_WINDOWS_PER_ITINERARY=2,
              BUYTIME_HISTORY_LENGTH=20,BUYTIME_BATCH_SIZE=4,FUTURE_HORIZON_VALUES=[2],
              bt_extra_stats=STATS,buytime_bounds=bundle['bounds'],TABULAR_Q_MAX_TRAIN_EPISODES=None,
              buytime_bundles={2:bundle},tabular_q_data={2:data},
              train_date_boundary=date(2022,1,10),stack_date_boundary=date(2022,1,20),
              val_date_boundary=date(2022,1,30))
    def frame(episodes):
        records=[e.record for e in episodes]
        keys=[r.key for r in records]
        X=np.stack([r.history[-1].numpy() for r in records])
        y=np.array([float(r.future[0]) for r in records],dtype=np.float32)
        anchors=np.array([r.current_log_price for r in records])
        f=pd.DataFrame(keys,columns=ns['KEY_COLUMNS'])
        f['features']=[cache.ArrayVector(row) for row in X]
        f['fareChange']=y;f['lagged_fare_1']=anchors
        return f,X,y,anchors,keys
    for split in ('train','val','test'):
        records=bundle['datasets'][split].records
        f,X,y,anchors,keys=frame(data['episodes'][split])
        for suffix in ('','_no_ts'):
            ns[f'X_{split}_np{suffix}']=X;ns[f'y_{split}_np{suffix}']=y
            name={'train':'xgb_train_pd','val':'xgb_val_pd','test':'common_test_pd'}[split]
            ns[name+suffix]=f
        ns[f'X_{split}_list_dl']=[r.history for r in records]
        ns[f'y_{split}_tensor_dl']=torch.from_numpy(y[:,None].copy())/.02
        ns[f'{split}_anchor_np']=anchors;ns[f'lengths_{split}']=[len(r.history) for r in records]
        ns[f'{split}_sequence_keys']=keys
    f,X,y,anchors,keys=frame(data['episodes']['calibration'])
    ns['prepare_ordered_sequences']=lambda *_:None
    ns['spark_sequences_to_variable_numpy']=lambda *_:([e.record.history for e in data['episodes']['calibration']],
                                                    torch.from_numpy(y[:,None].copy()),anchors,
                                                    [len(e.record.history) for e in data['episodes']['calibration']],keys)
    ns['aligned_tabular_rows']=lambda *_:f
    ns.update(df_processed_dl_all=object(),stack_target_keys=object(),val_df_processed=object())
    return cache.snapshot_model_data(ns),ns


class DatasetCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_roundtrip_all_model_inputs_preserves_storage_and_episode_identity(self):
        state,_=snapshot_fixture()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cache'
            cache.save_cache(path,state,'fixture')
            self.assertTrue(path.is_file())
            ns=namespace()
            with patch.object(cache.torch,'load',wraps=cache.torch.load) as loader:
                loaded=cache.load_cache(path,'fixture',{name:ns[name] for name in cache.DATA_CLASSES})
                self.assertTrue(loader.call_args.kwargs['weights_only'])
            cache.restore_model_data(loaded,ns)
            np.testing.assert_array_equal(ns['X_train_np'],state['price']['train']['X'])
            np.testing.assert_array_equal(ns['y_test_tensor_dl'].numpy(),state['dl']['test']['y'].numpy())
            self.assertIsInstance(ns['test_sequence_keys'][0][2],date)
            a,b=ns['buytime_bundles'][2]['datasets']['train'].records[:2]
            self.assertEqual(a.history.untyped_storage().data_ptr(),b.history.untyped_storage().data_ptr())
            episodes=ns['tabular_q_data'][2]['episodes']['train']
            self.assertIs(episodes[0].record,a)
            self.assertIs(episodes[0].feature_matrix,episodes[1].feature_matrix)
            self.assertTrue(ns['DATASET_CACHE_LOADED'])
            self.assertEqual(list(ns['buytime_bundles'][2]['datasets']),list(state['buytime'][2]['records']))

    def test_numpy_scalars_and_dates_use_restricted_compatible_encoding(self):
        state=dict(values=[np.float64(1.2),np.int64(3),np.bool_(True),date(2022,1,1)])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cache';cache.save_cache(path,state,'spec')
            self.assertEqual(cache.load_cache(path,'spec',{})['values'],[1.2,3,True,date(2022,1,1)])
        with self.assertRaises(TypeError):cache._encode(torch.nn.Linear(2,1))

    def test_legacy_directory_loads_and_checks_integrity(self):
        import hashlib, zipfile
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'prepared.pt';path.mkdir()
            payload=root/'datasets.pt';archive=root/'datasets.zip'
            torch.save(cache._encode(dict(a=np.arange(3))),payload)
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
                z.write(payload,'datasets.pt')
            part=path/'datasets-00000.part';part.write_bytes(archive.read_bytes())
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            manifest=dict(format_version=1,fingerprint='legacy',parts=[dict(name=part.name,bytes=part.stat().st_size,sha256=sha(part))],archive_sha256=sha(archive),payload_sha256=sha(payload),payload_bytes=payload.stat().st_size)
            (path/'manifest.json').write_text(json.dumps(manifest))
            loaded=cache.load_cache(path,'new',{},legacy_fingerprint='legacy')
            np.testing.assert_array_equal(loaded['a'],np.arange(3))
            with self.assertRaises(cache.CacheMismatch):cache.load_cache(path,'new',{})
            with self.assertRaises(IsADirectoryError):cache.save_cache(path,dict(a=1),'new',overwrite=True)
            part.write_bytes(b'broken')
            with self.assertRaises(cache.CacheMismatch):cache.load_cache(path,'new',{},legacy_fingerprint='legacy')

    def test_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cache.pt';cache.save_cache(path,dict(a=torch.arange(3)),'old')
            with self.assertRaises(cache.CacheMismatch):cache.load_cache(path,'new',{})

    def test_failed_save_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cache.pt';cache.save_cache(path,dict(a=1),'spec')
            with self.assertRaises(FileExistsError):cache.save_cache(path,dict(a=2),'spec')
            with patch.object(cache.torch,'save',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):cache.save_cache(path,dict(a=2),'spec',overwrite=True)
            self.assertEqual(cache.load_cache(path,'spec',{}),dict(a=1))
            cache.save_cache(path,dict(a=2),'spec',overwrite=True)
            self.assertEqual(cache.load_cache(path,'spec',{}),dict(a=2))

    def test_fingerprint_ignores_model_parameters_but_detects_window_changes(self):
        original=json.loads(NOTEBOOK.read_text())
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'notebook.ipynb';path.write_text(json.dumps(original))
            first=cache.preparation_fingerprint(path)
            changed=copy.deepcopy(original)
            c=next(c for c in changed['cells'] if ''.join(c['source']).startswith('# Fixed parameters and independent'))
            c['source']=''.join(c['source']).replace("'n_estimators': 500","'n_estimators': 501").splitlines(keepends=True)
            path.write_text(json.dumps(changed));self.assertEqual(first,cache.preparation_fingerprint(path))
            for c in changed['cells']:
                s=''.join(c['source'])
                if 'SEQUENCE_LENGTH = 10' in s:c['source']=s.replace('SEQUENCE_LENGTH = 10','SEQUENCE_LENGTH = 8').splitlines(keepends=True)
            path.write_text(json.dumps(changed));self.assertNotEqual(first,cache.preparation_fingerprint(path))
            self.assertNotEqual(first,cache.preparation_fingerprint(NOTEBOOK,'new-data-version'))

    def test_cache_hit_skips_all_raw_preparation_without_spark(self):
        from IPython.core.inputtransformer2 import TransformerManager
        state,_=snapshot_fixture();ns=namespace();cache.restore_model_data(state,ns)
        self.assertNotIn('spark',ns)
        transform=TransformerManager()
        with contextlib.redirect_stdout(io.StringIO()):
            for c in CELLS:
                s=''.join(c['source'])
                if '# DATASET-CACHE-GUARD' in s:exec(compile(transform.transform_cell(s),'guarded cell','exec'),ns)
        self.assertNotIn('spark',ns)
        self.assertNotIn('df_top5_mod',ns)

    def test_fresh_restored_runtime_trains_and_evaluates_every_model_family(self):
        state,_=snapshot_fixture();ns=namespace()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cache';cache.save_cache(path,state,'spec')
            state=cache.load_cache(path,'spec',{name:ns[name] for name in cache.DATA_CLASSES})
            cache.restore_model_data(state,ns)
            ns['best_xgboost_params_global']=dict(n_estimators=4,max_depth=2,learning_rate=.1)
            ns['best_xgboost_params_no_ts']=dict(n_estimators=4,max_depth=2,learning_rate=.1)
            ns['TABULAR_Q_XGB_PARAMS']=dict(ns['TABULAR_Q_XGB_PARAMS'],n_estimators=4,max_depth=2,min_child_weight=1,n_jobs=1,early_stopping_rounds=2)
            ns['BUYTIME_EPOCHS']=2;ns['BUYTIME_PATIENCE']=1
            original_cwd=Path.cwd()
            import os
            try:
                os.chdir(temp)
                with contextlib.redirect_stdout(io.StringIO()):
                    for c in CELLS:
                        s=''.join(c['source'])
                        if any(marker in s for marker in (
                            '# Optional baseline fit/evaluation;', '# Optional no-time-series fit/evaluation;',
                            '# Training diagnostics', 'class GRUModel(', 'class CNN1DModel(', 'class LSTMModel(',
                            'class TimeSeriesTransformer(', '# Persistence baseline on the same target rows:',
                            '# Keyed holdout stacking:', '# Run GRU independently', '# Run LSTM independently',
                            '# Run CNN independently', '# Run Dual-head GRU independently',
                            '# Fit the horizon regressors once;', '# Daily replanning: reuse', '# Fitted Q can run')):
                            s=s.replace('epochs = 100','epochs = 2').replace('epochs=100','epochs=2')
                            exec(compile(s,'cached model cell','exec'),ns)
            finally:os.chdir(original_cwd)
        self.assertNotIn('spark',ns)
        self.assertTrue(np.isfinite(ns['stacked_metrics']['skill_vs_persistence']))
        self.assertEqual(len(ns['buytime_results']),4)
        self.assertEqual(len(ns['tabular_sequential_results']),3)

if __name__=='__main__':unittest.main()
