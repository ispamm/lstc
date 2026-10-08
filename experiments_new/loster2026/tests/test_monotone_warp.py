import unittest,inspect
from dataclasses import replace
import numpy as np
from loster2026.config import Config
from loster2026.augmentation import augment,two_snapshots
from loster2026.monotone_warp import monotone_coordinates,augment_monotone
class MonotoneWarpTests(unittest.TestCase):
    def test_strict_finite_endpoints_and_lengths(self):
        rng=np.random.RandomState(918)
        for length in (2,3,60,96,427,470,512,1024,1500,1639):
            for _ in range(100):
                xp=monotone_coordinates(rng.normal(1,.2,6),length)
                self.assertEqual(len(xp),length);self.assertTrue(np.isfinite(xp).all())
                self.assertTrue(np.all(np.diff(xp)>0));self.assertEqual(xp[0],0);self.assertEqual(xp[-1],length-1)
    def test_zero_and_small_sigma_identity(self):
        rng=np.random.RandomState(14);z=rng.normal(size=6)
        for length in (2,60,1639):
            np.testing.assert_allclose(monotone_coordinates(np.ones(6),length),np.arange(length),rtol=0,atol=1e-12)
            np.testing.assert_allclose(monotone_coordinates(1+1e-10*z,length),np.arange(length),rtol=0,atol=1e-6)
    def test_label_free_signature(self):
        self.assertEqual(list(inspect.signature(augment_monotone).parameters),['series','config'])
        self.assertEqual(list(inspect.signature(monotone_coordinates).parameters),['multipliers','length'])
    def test_seed_replay_length_finite_and_rng_pairing(self):
        x=np.random.RandomState(18).normal(size=(32,96));c=replace(Config(),warp_mode='MonotoneWarp',augmentation_order=('sign','equal-segment-permutation','monotone-time-warp'))
        np.random.seed(4);a,b,info=two_snapshots(x,c);state=np.random.get_state()
        np.random.seed(4);a2,b2,_=two_snapshots(x,c)
        np.testing.assert_array_equal(a,a2);np.testing.assert_array_equal(b,b2)
        self.assertEqual(a.shape,x.shape);self.assertTrue(np.isfinite(a).all());self.assertGreater(info['A_train']['minimum_coordinate_difference'],0)
        np.random.seed(4);two_snapshots(x,Config());old=np.random.get_state()
        np.testing.assert_array_equal(state[1],old[1]);self.assertEqual(state[2:],old[2:])
    def test_no_silent_rescue(self):
        with self.assertRaises(ValueError):monotone_coordinates([np.nan]*6,60)
    def test_config_records_method_and_isolates_cache(self):
        old=Config();new=replace(old,warp_mode='MonotoneWarp',augmentation_order=('sign','equal-segment-permutation','monotone-time-warp'))
        new.validate();self.assertNotEqual(old.initialization_id(),new.initialization_id())
        with self.assertRaises(ValueError):replace(new,variant='LoSTer-NoResoftmax').validate()
