"""Run: python -m unittest discover -s tests -p test_pipeline.py"""
import sys,tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd,joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import simulate,train,score,features,MODEL_NAMES

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.data=simulate()
        cls.bundle,cls.scored,cls.report=train(cls.data,Path(cls.temp.name))
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_reproducible_data(self):
        pd.testing.assert_frame_equal(self.data,simulate())
        self.assertEqual(len(self.data),1200)
        self.assertEqual(self.data.id.nunique(),1200)
        self.assertEqual(self.data.late_days.isna().sum(),38)
    def test_chronological_split_and_train_only_imputation(self):
        self.assertEqual(self.report['split'],dict(training=670,calibration=134,test=396))
        medians=features(self.data[self.data.month<='2026-05']).median().to_numpy()
        np.testing.assert_allclose(self.bundle['imputer'].statistics_,medians)
        for values in self.bundle['calibration_scores'].values():self.assertEqual(len(values),134)
    def test_labels_excluded(self):
        self.assertEqual(list(features(self.data)),['cycle_ratio','variance_pct','change_orders','late_days'])
        other=self.data.copy();other.injected_anomaly=~other.injected_anomaly
        for name in MODEL_NAMES:
            np.testing.assert_allclose(score(self.data,self.bundle,name).anomaly_score,score(other,self.bundle,name).anomaly_score)
    def test_default_results(self):
        expected={'isolation_forest':(37,0,1,358),'one_class_svm':(21,10,17,348),'rules':(38,5,0,353)}
        for name,counts in expected.items():
            m=self.report['models'][name]
            self.assertEqual(tuple(m[k] for k in ['tp','fp','fn','tn']),counts)
    def test_saved_model_and_unlabeled_scoring(self):
        loaded=joblib.load(Path(self.temp.name)/'model_bundle.joblib')
        examples=self.data.tail(5).drop(columns=['injected_anomaly','quality_issue'])
        for name in MODEL_NAMES:
            np.testing.assert_allclose(score(examples,loaded,name).anomaly_score,score(examples,self.bundle,name).anomaly_score)
    def test_missingness_is_preserved(self):
        output=score(self.data,self.bundle)
        self.assertEqual(output.late_days.isna().sum(),38)
        self.assertEqual(output.imputed_late_days.isna().sum(),0)
    def test_threshold_monotonicity(self):
        for name in ['isolation_forest','one_class_svm']:
            loose=score(self.data,self.bundle,name,80)
            strict=score(self.data,self.bundle,name,99)
            self.assertTrue((~strict.flagged|loose.flagged).all())
    def test_rejects_bad_input(self):
        for col,val in [('planned_days',0),('budget_aed',0),('late_days',-1),('month','2026-99')]:
            bad=self.data.copy();bad.loc[0,col]=val
            with self.assertRaises(ValueError):features(bad)
        bad=self.data.copy();bad.loc[1,'id']=bad.loc[0,'id']
        with self.assertRaises(ValueError):features(bad)
        with self.assertRaises(ValueError):score(self.data,self.bundle,percentile=100)

if __name__=='__main__':unittest.main()
