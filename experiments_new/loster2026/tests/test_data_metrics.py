from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from support import available, legacy_nodes

@unittest.skipUnless(available("numpy"),"NumPy missing/unimportable")
class DataTests(unittest.TestCase):
    def test_zero_variance_nonfinite_rejected(self):
        import numpy as np
        from loster2026.data import normalize_rows
        for data in ([[1,1,1]],[[1,float("nan"),2]],[[1,float("inf"),2]]):
            with self.assertRaises(ValueError):normalize_rows(data)

    def test_population_z_normalization(self):
        import numpy as np
        from loster2026.data import normalize_rows
        x=normalize_rows([[1,2,3,4],[3,9,6,0]])
        np.testing.assert_allclose(x.mean(1),[0,0],atol=1e-15)
        np.testing.assert_allclose(x.std(1,ddof=0),[1,1],atol=1e-15)

    def test_headerless_all_rows_labels_and_k(self):
        from loster2026.data import load_ucr
        with tempfile.TemporaryDirectory(prefix="loster2026-header-") as root:
            base=Path(root)/"Tiny";base.mkdir()
            (base/"Tiny_TRAIN.tsv").write_text("-1\t1\t2\t3\t4\n2\t4\t1\t8\t2\n-1\t9\t7\t3\t4\n")
            (base/"Tiny_TEST.tsv").write_text("2\t1\t2\t5\t7\n-1\t1\t5\t3\t2\n")
            data=load_ucr(root,"Tiny",{"N":5,"L":4,"k":2,"train_count":3,"test_count":2})
            self.assertEqual(data.labels.tolist(),[0,1,0,1,0])
            self.assertEqual(data.x.shape,(5,4))
            self.assertEqual(data.metadata["oracle_k"],True)
            self.assertEqual([r["count"] for r in data.metadata["class_counts"]],[3,2])
            self.assertEqual(len(data.metadata["input_sha256"]["TRAIN"]),64)
            with self.assertRaises(ValueError):load_ucr(root,"Tiny",{"N":4})

    @unittest.skipUnless(available("pandas"),"pandas absent: literal historical header test unavailable")
    def test_legacy_parser_loses_first_row_new_keeps_all(self):
        import pandas as pd
        from loster2026.data import load_ucr
        with tempfile.TemporaryDirectory(prefix="loster2026-header-legacy-") as root:
            base=Path(root)/"Tiny";base.mkdir()
            for split in ("TRAIN","TEST"):
                (base/("Tiny_"+split+".tsv")).write_text("1\t1\t2\t3\n2\t4\t2\t7\n1\t7\t3\t1\n")
            legacy_train=pd.read_csv(base/"Tiny_TRAIN.tsv",sep="\t").values
            legacy_test=pd.read_csv(base/"Tiny_TEST.tsv",sep="\t").values
            self.assertEqual(len(legacy_train)+len(legacy_test),4)
            corrected=load_ucr(root,"Tiny")
            self.assertEqual(corrected.metadata["train_count"],3)
            self.assertEqual(corrected.metadata["test_count"],3)
            self.assertEqual(corrected.metadata["N"],6)
            self.assertEqual(corrected.k,2)

    def test_nonfinite_labels_rejected(self):
        from loster2026.data import load_ucr
        with tempfile.TemporaryDirectory() as root:
            base=Path(root)/"Tiny";base.mkdir()
            for split in ("TRAIN","TEST"):
                (base/("Tiny_"+split+".tsv")).write_text("nan\t1\t2\t3\n")
            with self.assertRaises(ValueError):load_ucr(root,"Tiny")

@unittest.skipUnless(available("numpy","scipy"),"NumPy/SciPy missing/unimportable")
class AugmentationAlignmentTests(unittest.TestCase):
    def test_augmentation_exact_legacy_semantics_and_rng(self):
        import numpy as np
        from loster2026.augmentation import augment
        from loster2026.config import Config
        ns=legacy_nodes("models/LoSTer/data/augmentation.py",
                        ["rotation","permutation","time_warp"],{"np":np})
        x=np.arange(16*24,dtype=float).reshape(16,24)
        np.random.seed(17)
        expected=ns["time_warp"](ns["permutation"](ns["rotation"](x[:,:,None]))).squeeze(2).astype(np.float32)
        expected_state=np.random.get_state()
        np.random.seed(17)
        actual,info=augment(x,Config())
        np.testing.assert_array_equal(actual,expected)
        actual_state=np.random.get_state()
        np.testing.assert_array_equal(actual_state[1],expected_state[1])
        self.assertEqual(actual_state[2:],expected_state[2:])
        self.assertIn("nonincreasing_step_fraction",info)

    def test_two_fixed_snapshots_replay_and_remain_distinct(self):
        import numpy as np
        from loster2026.augmentation import two_snapshots
        from loster2026.config import Config
        x=np.arange(16*24,dtype=float).reshape(16,24)
        np.random.seed(3);a,b,info=two_snapshots(x,Config())
        np.random.seed(3);a2,b2,_=two_snapshots(x,Config())
        np.testing.assert_array_equal(a,a2);np.testing.assert_array_equal(b,b2)
        self.assertNotEqual(info["A_train"]["snapshot_sha256"],info["A_init"]["snapshot_sha256"])

    def test_ragged_equal_segments_preserve_length(self):
        import numpy as np
        from loster2026.augmentation import augment
        from loster2026.config import Config
        np.random.seed(5)
        a,_=augment(np.arange(8*25,dtype=float).reshape(8,25),Config())
        self.assertEqual(a.shape,(8,25))

    def test_alignment_uses_memberships_not_coordinates(self):
        import numpy as np
        from loster2026.alignment import match_initial_centers
        original=np.array([0,0,1,1,2,2])
        augmented=np.array([2,2,0,0,1,1])
        centers=np.array([[900,0],[0,900],[-900,0]],dtype=np.float32)
        matched,info=match_initial_centers(original,augmented,centers)
        np.testing.assert_array_equal(matched,centers[[2,0,1]])
        self.assertEqual(info["same_index_agreement_after"],1.)
        self.assertFalse(info["ground_truth_used"])
        self.assertEqual(info["matching_calls"],1)

    def test_alignment_ties_replay_deterministically(self):
        import numpy as np
        from loster2026.alignment import match_initial_centers
        args=([0,0,1,1],[0,1,0,1],np.eye(2))
        a,x=match_initial_centers(*args);b,y=match_initial_centers(*args)
        np.testing.assert_array_equal(a,b);self.assertEqual(x["permutation"],y["permutation"])

    def test_hard_occupancy_exposes_empty_and_collapse(self):
        from loster2026.diagnostics import occupancy
        item=occupancy([0]*8,3,[0]*8)
        self.assertTrue(item["complete_collapse"])
        self.assertEqual(item["minimum_cluster_size"],0)
        self.assertEqual(item["hard_entropy"],0.)
        self.assertEqual(item["assignment_change_fraction"],0.)
        self.assertEqual(item["effective_cluster_count"],1.)

@unittest.skipUnless(available("numpy","scipy","sklearn"),"NumPy/SciPy/sklearn missing/unimportable")
class MetricTests(unittest.TestCase):
    def test_perfect_permutation(self):
        from loster2026.metrics import evaluate
        self.assertEqual(evaluate([0,0,1,1],[8,8,3,3]),{"ARI":1.,"NMI_arithmetic":1.,"RI":1.,"ACC":1.})

    def test_known_cross_partition(self):
        from loster2026.metrics import evaluate
        result=evaluate([0,0,1,1],[0,1,0,1])
        self.assertAlmostEqual(result["ARI"],-.5,14)
        self.assertAlmostEqual(result["NMI_arithmetic"],0.,14)
        self.assertAlmostEqual(result["RI"],1/3,14)
        self.assertEqual(result["ACC"],.5)

    def test_single_predicted_cluster(self):
        from loster2026.metrics import evaluate
        result=evaluate([0,0,1,1],[9,9,9,9])
        self.assertEqual(result["ARI"],0.)
        self.assertEqual(result["NMI_arithmetic"],0.)
        self.assertAlmostEqual(result["RI"],1/3,14)
        self.assertEqual(result["ACC"],.5)

    def test_explicit_arithmetic_not_geometric(self):
        from loster2026.metrics import evaluate
        from sklearn.metrics import normalized_mutual_info_score
        y=[0,0,0,1,1,1];p=[0,0,1,2,2,2]
        value=evaluate(y,p)["NMI_arithmetic"]
        self.assertEqual(value,normalized_mutual_info_score(y,p,average_method="arithmetic"))
        self.assertNotAlmostEqual(value,normalized_mutual_info_score(y,p,average_method="geometric"),8)
