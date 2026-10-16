"""Regression checks for the notebook's buy-time experiment (no airfare parquet needed)."""
import copy
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
import numpy as np
import torch

NOTEBOOK = Path(__file__).resolve().parents[1] / 'Gabor_Galovics_Thesis_work_3_FIXED(2).ipynb'
notebook = json.loads(NOTEBOOK.read_text())
source = next(''.join(c['source']) for c in notebook['cells']
              if ''.join(c['source']).startswith('# Buy-time helpers:'))
bt = ModuleType('notebook_buytime_helpers')
sys.modules[bt.__name__] = bt
exec(compile(source, str(NOTEBOOK), 'exec'), bt.__dict__)

def observations(leg='a', missing=(), no_features=(), level=4.5, trend=0.02):
    start = date(2022, 1, 1)
    steps = [SimpleNamespace(searchDate=start + timedelta(days=i),
                             log_price=level + trend*i, change=trend,
                             features=None if i in no_features else np.array([i/40, 1.0]))
             for i in range(40) if i not in missing]
    return SimpleNamespace(legId=leg, flightDate=date(2022, 4, 1), observations=steps)

START = date(2022, 1, 1)
BOUNDS = {'train':(None, START+timedelta(days=9)),
          'val':(START+timedelta(days=9), START+timedelta(days=19)),
          'calibration':(START+timedelta(days=19), START+timedelta(days=29)),
          'test':(START+timedelta(days=29), START+timedelta(days=39))}
STATS = {'price_mean':4.5,'price_std':1.0,'change_mean':0.0,'change_std':1.0}

def build(rows=None):
    return bt.make_buytime_records(rows or [observations()], 2, BOUNDS, STATS,
                                  history_length=20,min_history=3,max_train_origins=3)

class BuyTimeDataTests(unittest.TestCase):
    def test_all_labels_are_daily_and_purged_by_origin(self):
        records, coverage = build()
        self.assertEqual([len(records[k]) for k in BOUNDS], [3,1,1,1])
        all_keys = []
        label_sets = []
        for split, values in records.items():
            lower, upper = BOUNDS[split]
            labels = set()
            for r in values:
                self.assertLessEqual(r.key[2], upper)
                if lower is not None: self.assertGreater(r.key[2], lower)
                self.assertEqual(r.history_dates[-1], r.key[2])
                self.assertTrue(all(day <= r.key[2] for day in r.history_dates))
                self.assertEqual(r.future_dates, (r.key[2]+timedelta(days=1), r.key[2]+timedelta(days=2)))
                self.assertLessEqual(r.future_dates[-1], upper)
                self.assertLess(r.future_dates[-1], r.key[1])
                self.assertLessEqual(len(r.history), 20)
                self.assertTrue(torch.allclose(r.future, torch.tensor([.02,.04]), atol=1e-6))
                labels.update(r.future_dates); all_keys.append(r.key)
            label_sets.append(labels)
        self.assertEqual(len(all_keys),len(set(all_keys)))
        for i in range(len(label_sets)):
            for j in range(i): self.assertFalse(label_sets[i] & label_sets[j])
        self.assertGreater(coverage['train']['boundary_purged'],0)
        for counts in coverage.values():
            accounted = sum(counts[key] for key in ('no_features','short_history','boundary_purged','departure_purged','missing_followup','complete_eligible'))
            self.assertEqual(counts['candidates'],accounted)
            self.assertEqual(counts['complete_eligible'],counts['selected']+counts['capped_out'])

    def test_gaps_are_not_silently_relabeled_as_tomorrow(self):
        records, coverage = build([observations(missing=(18,), no_features=(17,))])
        self.assertGreater(coverage['val']['missing_followup'],0)
        r = records['val'][0]
        self.assertEqual(r.key[2], START+timedelta(days=15))
        self.assertEqual(r.future_dates[-1], START+timedelta(days=17))
        self.assertTrue(torch.allclose(r.future, torch.tensor([.02,.04]), atol=1e-6))

    def test_future_features_cannot_change_past_input(self):
        original = observations()
        changed = copy.deepcopy(original)
        for step in changed.observations:
            if step.searchDate > START+timedelta(days=2): step.features[:] = 999999
        first = build([original])[0]['train'][0]
        second = build([changed])[0]['train'][0]
        self.assertEqual(first.key,second.key)
        self.assertTrue(torch.equal(first.history,second.history))

    def test_zero_log_price_is_valid_not_padding(self):
        records, _ = build([observations(level=0.0,trend=0.0)])
        self.assertTrue(records['train'])
        self.assertTrue(torch.equal(records['train'][0].future,torch.zeros(2)))
        bundle = bt.make_buytime_bundle(records, batch_size=8)
        self.assertTrue(torch.equal(bundle['scales'],torch.ones(2)))

    def test_duplicate_groups_or_dates_are_rejected(self):
        row = observations()
        with self.assertRaises(ValueError): build([row,row])
        row.observations.append(copy.deepcopy(row.observations[0]))
        with self.assertRaises(ValueError): build([row])

    def test_empty_split_requires_explicit_action(self):
        records, _ = build()
        records['calibration'] = []
        with self.assertRaisesRegex(ValueError,'calibration'): bt.make_buytime_bundle(records)

class BuyTimeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_mask_does_not_create_synthetic_zero_labels(self):
        p=torch.tensor([[2.,999.]],requires_grad=True)
        y=torch.zeros_like(p)
        loss=bt.masked_mse_loss(p,y,torch.tensor([[1.,0.]]))
        self.assertEqual(float(loss.detach()),4.)
        loss.backward(); self.assertEqual(float(p.grad[0,1]),0.)
        with self.assertRaises(ValueError): bt.masked_mse_loss(p,y,torch.zeros_like(p))

    def test_padding_and_batch_sort_order_do_not_change_predictions(self):
        torch.manual_seed(42)
        x=torch.randn(2,5,4); lengths=torch.tensor([2,5])  # deliberately unsorted
        altered=x.clone(); altered[0,2:]=100000
        for kind in ('GRU','LSTM','CNN'):
            model=bt.BuyTimeTrajectoryModel(4,2,kind,hidden_size=8,num_layers=1,dropout=0.)
            # Zero head would make this check trivial, so give the decoder nonzero weights.
            with torch.no_grad(): model.regression_head.weight.normal_(0,.2)
            model.eval()
            self.assertTrue(torch.allclose(model(x,lengths),model(altered,lengths),atol=1e-6),kind)
            alone=model(x[:1,:2],lengths[:1])
            self.assertTrue(torch.allclose(model(x,lengths)[:1],alone,atol=1e-6),kind)

    def test_all_architectures_can_learn_tiny_batch(self):
        records, _=build([observations(leg=str(i),level=4.5+i/10) for i in range(3)])
        bundle=bt.make_buytime_bundle(records,batch_size=32)
        for kind,dual in [('GRU',False),('LSTM',False),('CNN',False),('GRU',True)]:
            torch.manual_seed(42)
            model=bt.BuyTimeTrajectoryModel(bundle['feature_dim'],2,kind,hidden_size=8,num_layers=1,dropout=0.,dual=dual)
            initial=bt.buytime_epoch(model,bundle['loaders']['train'],bundle['scales'],bundle['train_mean_price'])['loss']
            opt=torch.optim.Adam(model.parameters(),lr=.02)
            for _ in range(15): bt.buytime_epoch(model,bundle['loaders']['train'],bundle['scales'],bundle['train_mean_price'],opt)
            final=bt.buytime_epoch(model,bundle['loaders']['train'],bundle['scales'],bundle['train_mean_price'])['loss']
            self.assertLess(final,.7*initial,(kind,dual,initial,final))

    def test_complete_training_calibration_and_evaluation(self):
        records,_=build([observations(leg=str(i),level=4.5+i/10) for i in range(3)])
        bundle=bt.make_buytime_bundle(records,batch_size=32)
        for kind,dual in [('GRU',False),('LSTM',False),('CNN',False),('GRU',True)]:
            result=bt.run_buytime_experiment(kind,bundle,dual=dual,epochs=2,patience=1,
                                             hidden_size=8,num_layers=1,dropout=0.,device=torch.device('cpu'))
            self.assertEqual(len(result['history']),2)
            self.assertTrue(np.isfinite(result['metrics']['avg_money_saved']))
            self.assertGreaterEqual(result['policy']['calibration_savings'],-1e-10)
        with self.assertRaises(ValueError):
            bt.predict_buytime(result['model'],bundle['loaders']['train'],bundle['scales'])

class BuyTimeFinancialTests(unittest.TestCase):
    def test_savings_use_actual_selected_price_and_include_buy_now(self):
        anchors=np.log([100.,100.,100.]);truth=np.log([[.9,1.1],[1.,1.2],[1.1,1.2]])
        result=bt.purchase_costs(anchors,truth,np.array([2,1,0]))
        np.testing.assert_allclose(result['savings'],[-10.,0.,0.],atol=1e-10)
        np.testing.assert_allclose(result['oracle_savings'],[10.,0.,0.],atol=1e-10)
        np.testing.assert_allclose(result['regret'],[20.,0.,0.],atol=1e-10)
        np.testing.assert_array_equal(result['optimal'],[False,True,True])
        with self.assertRaises(ValueError): bt.purchase_costs(anchors,truth,np.array([3,0,0]))

    def test_bad_calibration_predictions_select_buy_now_fallback(self):
        data=dict(prediction=np.log([[.9,.95],[.9,.95]]),truth=np.log([[1.1,1.2],[1.1,1.2]]),
                  anchors=np.log([100.,100.]),logits=None,keys=[('a',date(2022,4,1),START)]*2)
        policy=bt.calibrate_buytime_policy(data,np.ones(2))
        self.assertTrue(np.isinf(policy['threshold']))
        self.assertEqual(policy['calibration_savings'],0.)
        metrics=bt.evaluate_buytime_policy(data,np.ones(2),policy)
        self.assertEqual(metrics['avg_money_saved'],0.)
        self.assertLess(metrics['raw_policy_money_saved'],0.)
        with self.assertRaises(ValueError): bt.calibrate_buytime_policy(data,np.ones(2),thresholds=(0.,1.))

    def test_dual_action_head_is_actually_used(self):
        actions,gains=bt.proposed_buytime_actions(np.log([[.9,.8]]),np.log([100.]),np.ones(2),logits=np.array([[0.,10.,-10.]]))
        self.assertEqual(int(actions[0]),1)
        self.assertAlmostEqual(float(gains[0]),10.)
        self.assertEqual(int(bt.gated_buytime_actions(actions,gains,20.)[0]),0)

    def test_dollar_regret_penalizes_costly_wrong_actions(self):
        targets=torch.log(torch.tensor([[.9,1.5]]));scales=torch.ones(2);masks=torch.ones_like(targets)
        good=torch.tensor([[0.,10.,-10.]])
        bad=torch.tensor([[0.,-10.,10.]])
        y=torch.zeros_like(targets)
        good_loss,_=bt.buytime_objective((y,good),targets,masks,scales,torch.log(torch.tensor([100.])),100.)
        bad_loss,_=bt.buytime_objective((y,bad),targets,masks,scales,torch.log(torch.tensor([100.])),100.)
        self.assertGreater(float(bad_loss),float(good_loss))

    def test_financial_term_distinguishes_same_class_but_different_dollar_errors(self):
        logits=torch.tensor([[0.,-10.,10.]])  # same confidently wrong day in both cases
        low=torch.log(torch.tensor([[.9,1.1]]))
        high=torch.log(torch.tensor([[.9,1.8]]))
        masks=torch.ones_like(low);scales=torch.ones(2)
        current=torch.log(torch.tensor([100.]))
        low_loss,_=bt.buytime_objective((low.clone(),logits),low,masks,scales,current,100.)
        high_loss,_=bt.buytime_objective((high.clone(),logits),high,masks,scales,current,100.)
        # Identical class labels, logits and zero regression errors: CE cannot explain this gap.
        self.assertGreater(float(high_loss-low_loss),.69)

    def test_frozen_policy_can_lose_on_test_without_retuning_to_buy_now(self):
        calibration=dict(prediction=np.log([[.8,.9]]),truth=np.log([[.8,.9]]),
                         anchors=np.log([100.]),logits=None,
                         keys=[('a',date(2022,4,1),START)])
        policy=bt.calibrate_buytime_policy(calibration,np.ones(2))
        frozen=copy.deepcopy(policy)
        holdout=copy.deepcopy(calibration)
        holdout['truth']=np.log([[1.2,1.3]])
        metrics=bt.evaluate_buytime_policy(holdout,np.ones(2),policy)
        self.assertLess(metrics['avg_money_saved'],0.)
        self.assertEqual(metrics['wait_fraction'],1.)
        self.assertEqual(policy,frozen)

    def test_smearing_corrects_log_to_dollar_bias(self):
        data=dict(prediction=np.zeros((2,2)),truth=np.log([[2.,3.],[2.,3.]]))
        np.testing.assert_allclose(bt.fit_smearing_factor(data),[2.,3.])

if __name__=='__main__': unittest.main()
