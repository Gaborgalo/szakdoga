"""Notebook regression tests for appended forecast/replanning/fitted-Q experiments."""
import copy
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

NOTEBOOK = Path(__file__).resolve().parents[1] / 'Gabor_Galovics_Thesis_work_3_FIXED(2).ipynb'
notebook = json.loads(NOTEBOOK.read_text())
tq = ModuleType('notebook_tabular_q_helpers')
sys.modules[tq.__name__] = tq
for marker in ('# Buy-time helpers:', '# Appended XGBoost / fitted-Q helpers.'):
    source = next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith(marker))
    exec(compile(source, str(NOTEBOOK), 'exec'), tq.__dict__)

START = date(2022,1,1)
BOUNDS = {'train':(None,START+timedelta(days=9)),
          'val':(START+timedelta(days=9),START+timedelta(days=19)),
          'calibration':(START+timedelta(days=19),START+timedelta(days=29)),
          'test':(START+timedelta(days=29),START+timedelta(days=39))}
STATS = dict(price_mean=4.5,price_std=1.,change_mean=0.,change_std=1.)

def fixture_rows(missing_features=(), trend=.02):
    return [SimpleNamespace(legId=str(leg),flightDate=date(2022,4,1),observations=[
        SimpleNamespace(searchDate=START+timedelta(days=day), log_price=4.5+leg/10+trend*day,
                        change=trend,features=None if day in missing_features else np.array([day/40,leg/10]))
        for day in range(40)]) for leg in range(3)]

def fixture(missing_features=(),trend=.02):
    rows=fixture_rows(missing_features,trend)
    records,_=tq.make_buytime_records(rows,2,BOUNDS,STATS,history_length=20,min_history=3,max_train_origins=3)
    bundle=tq.make_buytime_bundle(records,batch_size=16)
    bundle['bounds']=BOUNDS
    data=tq.prepare_tabular_q_data(rows,bundle,STATS,history_limit=20)
    return rows,bundle,data

class ConstantRegressor:
    def __init__(self,value): self.value=value;self.calls=0
    def predict(self,X): self.calls+=1;return np.full(len(X),self.value,dtype=float)
    def fit(self,*args,**kwargs): raise AssertionError('Daily replay must not refit models')

class TabularQDataTests(unittest.TestCase):
    def test_same_original_keys_labels_and_model_free_preparation(self):
        rows,bundle,data=fixture()
        with patch.object(tq.xgb,'XGBRegressor',side_effect=AssertionError('Preparation must not fit XGBoost')):
            data=tq.prepare_tabular_q_data(rows,bundle,STATS,history_limit=20)
        for split,ds in bundle['datasets'].items():
            episodes=data['episodes'][split]
            self.assertEqual([e.record.key for e in episodes],[r.key for r in ds.records])
            got=tq.tabular_q_prediction_data(episodes)
            np.testing.assert_allclose(got['truth'],np.stack([r.future.numpy() for r in ds.records]))
            self.assertEqual(got['keys'],[r.key for r in ds.records])
        self.assertEqual(data['original_train_origins'],len(data['episodes']['train']))

    def test_training_cap_does_not_change_any_heldout_cohort(self):
        rows,bundle,full=fixture()
        capped=tq.prepare_tabular_q_data(rows,bundle,STATS,20,max_train_episodes=2)
        self.assertEqual(len(capped['episodes']['train']),2)
        self.assertEqual(capped['original_train_origins'],9)
        for split in ('val','calibration','test'):
            self.assertEqual([e.record.key for e in capped['episodes'][split]],
                             [e.record.key for e in full['episodes'][split]])

    def test_only_observed_covariates_enter_current_state(self):
        rows,bundle,data=fixture()
        changed=copy.deepcopy(rows)
        for row in changed:
            for observation in row.observations:
                if observation.searchDate>=START+timedelta(days=38): observation.features[:]=99999
        altered=tq.prepare_tabular_q_data(changed,bundle,STATS,20)
        for before,after in zip(data['episodes']['test'],altered['episodes']['test']):
            np.testing.assert_array_equal(before.state(0),after.state(0))
            self.assertFalse(np.array_equal(before.state(1),after.state(1)))
            with self.assertRaises(ValueError): before.state(3)

    def test_missing_future_covariates_are_causally_carried_forward(self):
        _,_,data=fixture(missing_features=(38,))
        for episode in data['episodes']['test']:
            self.assertEqual(episode.record.key[2],START+timedelta(days=37))
            self.assertEqual(episode.state(1)[-4],1.)  # age of last available covariates
            self.assertEqual(episode.state(2)[-4],0.)
            self.assertAlmostEqual(float(episode.state(1)[-2]),float(episode.observed_logs[1]-STATS['price_mean']),places=5)
            self.assertTrue(np.isfinite(episode.state(1)).all())

    def test_bad_episode_coverage_or_split_dates_are_rejected(self):
        rows,bundle,_=fixture()
        changed=copy.deepcopy(rows)
        changed[0].observations=[o for o in changed[0].observations if o.searchDate!=START+timedelta(days=38)]
        with self.assertRaises(ValueError): tq.prepare_tabular_q_data(changed,bundle,STATS,20)
        bundle['bounds']['test']=(BOUNDS['test'][0],START+timedelta(days=38))
        try:
            with self.assertRaises(ValueError): tq.prepare_tabular_q_data(rows,bundle,STATS,20)
        finally:
            BOUNDS['test']=(START+timedelta(days=29),START+timedelta(days=39))

class BellmanAndReplayTests(unittest.TestCase):
    def test_bellman_backup_uses_predicted_value_not_realized_oracle(self):
        np.testing.assert_allclose(tq.fitted_q_backup_targets([100.],[120.]),[.2])
        # Even if an unobserved later price were $50, a predicted next wait cost of +10%
        # implies buying at $120 next; the backup cannot choose that actual future $50.
        np.testing.assert_allclose(tq.fitted_q_backup_targets([100.],[120.],[.1]),[.2])
        np.testing.assert_allclose(tq.fitted_q_backup_targets([100.],[120.],[-.5]),[-.4])
        np.testing.assert_allclose(tq.fitted_q_backup_targets([100.],[120.],[-2.]),[-1.])
        with self.assertRaises(ValueError): tq.fitted_q_backup_targets([0.],[120.])
        with self.assertRaises(ValueError): tq.fitted_q_backup_targets([100.],[120.],[np.nan])

    def test_replay_uses_only_current_day_and_forces_buying_at_original_deadline(self):
        gains=np.array([[0.,999.,999.],[5.,0.,999.],[5.,5.,5.],[5.,5.,0.]])
        np.testing.assert_array_equal(tq.rollout_wait_gains(gains,0.),[0,1,3,2])
        np.testing.assert_array_equal(tq.rollout_wait_gains(gains,np.inf),[0,0,0,0])
        with self.assertRaises(ValueError): tq.rollout_wait_gains(gains,-1.)

    def test_replanning_reuses_models_and_shrinks_forecast_horizon(self):
        _,_,data=fixture()
        episodes=data['episodes']['test']
        models=[ConstantRegressor(np.log(1.05)),ConstantRegressor(np.log(.8))]
        raw,corrected=tq.rolling_forecast_gains(episodes,models,np.ones(2))
        current=np.exp([e.observed_logs[0] for e in episodes])
        np.testing.assert_allclose(raw[:,0],.2*current)
        np.testing.assert_allclose(raw[:,1],0.)  # second-day decision can only forecast tomorrow
        np.testing.assert_allclose(raw,corrected)
        self.assertEqual([m.calls for m in models],[2,1])
        np.testing.assert_array_equal(tq.rollout_wait_gains(raw,0.),np.ones(len(episodes)))
        altered=copy.deepcopy(episodes)
        for episode in altered: episode.observed_logs[1:]+=1.
        changed,_=tq.rolling_forecast_gains(altered,models,np.ones(2))
        np.testing.assert_allclose(changed[:,0],raw[:,0])

    def test_q_actions_use_the_remaining_time_model(self):
        _,_,data=fixture()
        episodes=data['episodes']['test']
        models={2:ConstantRegressor(-.1),1:ConstantRegressor(.2)}
        gains=tq.fitted_q_wait_gains(episodes,models)
        self.assertTrue(np.all(gains[:,0]>0))
        self.assertTrue(np.all(gains[:,1]<0))
        np.testing.assert_array_equal(tq.rollout_wait_gains(gains,0.),np.ones(len(episodes)))
        self.assertEqual([models[r].calls for r in (2,1)],[1,1])

    def test_calibration_can_choose_buy_now_but_cannot_guarantee_test_profit(self):
        keys=[('a',date(2022,4,1),START)]
        cal=dict(anchors=np.log([100.]),truth=np.log([[.9,.8]]),keys=keys)
        gains=np.array([[15.,15.]])
        policy=tq.calibrate_sequential_gate(cal,gains)
        self.assertGreater(policy['calibration_savings'],0.)
        frozen=copy.deepcopy(policy)
        test=dict(anchors=np.log([100.]),truth=np.log([[1.1,1.2]]),keys=keys)
        metrics=tq.summarize_sequential_policy(test,gains,gains,policy)
        self.assertLess(metrics['avg_money_saved'],0.)
        self.assertEqual(policy,frozen)
        fallback=tq.calibrate_sequential_gate(test,gains)
        self.assertTrue(np.isinf(fallback['threshold']))
        self.assertEqual(tq.summarize_sequential_policy(test,gains,gains,fallback)['avg_money_saved'],0.)
        with self.assertRaises(ValueError): tq.calibrate_sequential_gate(test,gains,thresholds=(0.,1.))

class TabularQFitTests(unittest.TestCase):
    def test_real_xgboost_forecasts_replanning_and_fitted_q_end_to_end(self):
        _,_,data=fixture(trend=-.02)
        params=dict(tq.TABULAR_Q_XGB_PARAMS)
        params.update(n_estimators=8,max_depth=2,min_child_weight=1,n_jobs=1,
                      early_stopping_rounds=3,subsample=1.,colsample_bytree=1.)
        fitted=tq.fit_buytime_xgboost(data,params)
        identities=[id(m.estimator) for m in fitted['models']]
        fixed=tq.evaluate_fixed_buytime_xgboost(data,fitted)
        rolling=tq.run_rolling_buytime_xgboost(data,fitted)
        self.assertEqual([id(m.estimator) for m in fitted['models']],identities)
        q=tq.fit_buytime_fitted_q(data,params)
        q_metrics=tq.evaluate_buytime_fitted_q(data,q)
        self.assertEqual(len(fitted['models']),2)
        self.assertEqual(set(q['models']),{1,2})
        for metrics in (fixed,rolling['metrics'],q_metrics):
            self.assertTrue(np.isfinite(metrics['avg_money_saved']))
            self.assertGreater(metrics['raw_policy_money_saved'],0.)
            self.assertAlmostEqual(metrics['oracle_money_saved'],fixed['oracle_money_saved'])
            self.assertGreaterEqual(metrics['mean_purchase_day'] if 'mean_purchase_day' in metrics else 0.,0.)
        self.assertGreaterEqual(fitted['policy']['calibration_savings'],0.)
        self.assertGreaterEqual(rolling['policy']['calibration_savings'],0.)
        self.assertGreaterEqual(q['policy']['calibration_savings'],0.)

if __name__=='__main__': unittest.main()
