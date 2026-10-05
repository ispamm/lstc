from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from support import available, legacy_nodes

TORCH = available("numpy","torch")

@unittest.skipUnless(TORCH,"NumPy/PyTorch missing or unimportable")
class LegacyMathematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from torch import nn
        from torch.nn import functional as F
        cls.torch=torch
        cls.arch=legacy_nodes("models/LoSTer/net/model.py",
                             ["ResidualBlock","PredictionResidualBlock","AE"],{"torch":torch,"nn":nn})
        cls.loss=legacy_nodes("models/LoSTer/utils/losses.py",
                             ["KMeansLoss","InstanceContrastiveLoss","ClusterContrastiveLoss"],
                             {"torch":torch,"nn":nn,"F":F})
        cls.assign=legacy_nodes("models/LoSTer/utils/gumbel.py",["softmax_logits"],{"torch":torch,"F":F})

    def assert_tensor(self,actual,expected,double=False):
        t=self.torch
        self.assertTrue(t.allclose(actual,expected,rtol=1e-10 if double else 1e-6,
                                  atol=1e-12 if double else 1e-7),
                        "Numerical regression mismatch")

    def test_residual_formula_and_dropout(self):
        import torch
        from torch.nn import functional as F
        from loster2026.architecture import ResidualBlock
        block=ResidualBlock(5,7,6,dropout=.1)
        x=torch.randn(3,5)
        torch.manual_seed(7)
        middle=F.relu(F.linear(x,block.block[0].weight,block.block[0].bias))
        hidden=F.linear(middle,block.block[2].weight,block.block[2].bias)
        expected=F.layer_norm(F.dropout(hidden,p=.1,training=True)+block.residual(x),
                              [6],block.layer_norm.weight,block.layer_norm.bias,1e-5)
        torch.manual_seed(7)
        self.assert_tensor(block(x),expected)

    def test_residual_block_matches_actual_legacy_state(self):
        import torch
        from loster2026.architecture import ResidualBlock
        ref=self.arch["ResidualBlock"](5,7,6,.1)
        new=ResidualBlock(5,7,6,.1);new.load_state_dict(ref.state_dict())
        ref.eval();new.eval()
        x=torch.randn(4,5);self.assert_tensor(new(x),ref(x))

    def test_prediction_block_has_no_dropout_or_normalization(self):
        import torch
        from loster2026.architecture import PredictionResidualBlock
        ref=self.arch["PredictionResidualBlock"](5,7,6)
        new=PredictionResidualBlock(5,7,6);new.load_state_dict(ref.state_dict())
        x=torch.randn(4,5);self.assert_tensor(new(x),ref(x))
        self.assertFalse(any(isinstance(m,(torch.nn.Dropout,torch.nn.LayerNorm)) for m in new.modules()))

    def test_encoder_decoder_shapes_and_numerical_equivalence(self):
        import torch
        from loster2026.architecture import Autoencoder
        from loster2026.config import Config
        c=replace(Config(),latent_dim=8)
        ref=self.arch["AE"](20,20,8,3,3,.1,False,False)
        new=Autoencoder(20,c);new.load_state_dict(ref.state_dict())
        ref.eval();new.eval();x=torch.randn(3,20,1)
        r,z=new(x);rr,zr=ref(x)
        self.assertEqual(tuple(r.shape),(3,20,1))
        self.assertEqual(tuple(z.shape),(3,8))
        self.assert_tensor(r,rr);self.assert_tensor(z,zr)

    def test_independent_view_weights(self):
        from loster2026.architecture import TwoViewModel
        from loster2026.config import Config
        model=TwoViewModel(10,3,replace(Config(),latent_dim=8))
        a=next(model.original.parameters());b=next(model.augmented.parameters())
        self.assertNotEqual(a.data_ptr(),b.data_ptr())
        self.assertFalse(self.torch.equal(a,b))

    def test_rbf_logits_and_sigma1(self):
        import torch
        from loster2026.assignments import rbf_logits
        z=torch.randn(4,7,dtype=torch.float64);c=torch.randn(3,7,dtype=torch.float64)
        for sigma in (1.,2.3):
            self.assert_tensor(rbf_logits(z,c,sigma),self.assign["softmax_logits"](z,c,sigma),True)
        self.assert_tensor(rbf_logits(z,c),rbf_logits(z,c,1.),True)

    def test_deterministic_nearest_centroid_and_ties(self):
        import torch
        from loster2026.assignments import deterministic_assignments
        z=torch.tensor([[0.,0.],[5.,0.],[2.5,0.]])
        c=torch.tensor([[0.,0.],[5.,0.]])
        self.assertEqual(deterministic_assignments(z,c).tolist(),[0,1,0])

    def test_hard_gumbel_fixed_seed_and_backward(self):
        import torch
        from torch.nn import functional as F
        from loster2026.assignments import hard_assignments
        logits=torch.randn(4,3,requires_grad=True)
        torch.manual_seed(41);actual=hard_assignments(logits,2.)
        torch.manual_seed(41);expected=F.gumbel_softmax(logits,tau=2.,hard=True)
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(torch.equal(actual.sum(1),torch.ones(4)))
        (actual*torch.arange(3.)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(),0)

    def test_kmeans_loss_matches_legacy(self):
        import torch
        from loster2026.losses import kmeans_loss
        z=torch.randn(4,5,dtype=torch.float64)
        c=torch.randn(3,5,dtype=torch.float64)
        q=torch.nn.functional.one_hot(torch.tensor([0,1,2,1]),3).double()
        ref=self.loss["KMeansLoss"](c.numpy())
        self.assert_tensor(kmeans_loss(z,q,c),ref(z,q),True)

    def test_instance_loss_matches_legacy(self):
        import torch
        from loster2026.config import Config
        from loster2026.losses import instance_loss
        a=torch.randn(4,5,dtype=torch.float64);b=torch.randn(4,5,dtype=torch.float64)
        self.assert_tensor(instance_loss(a,b,Config()),self.loss["InstanceContrastiveLoss"]()(a,b,1.),True)

    def test_cluster_contrast_entropy_and_extra_softmax(self):
        import torch
        from loster2026.config import Config
        from loster2026.losses import cluster_terms,cluster_representation
        q=torch.nn.functional.one_hot(torch.tensor([0,1,2,1]),4).double().requires_grad_()
        qa=torch.nn.functional.one_hot(torch.tensor([1,1,0,2]),4).double().requires_grad_()
        contrast,entropy=cluster_terms(q,qa,Config())
        ref=self.loss["ClusterContrastiveLoss"]()(q,qa,1.)
        self.assert_tensor(contrast+entropy,ref,True)
        s=q.softmax(1);sa=qa.softmax(1)
        p=s.sum(0)/s.sum();pa=sa.sum(0)/sa.sum()
        self.assert_tensor(entropy,(p*p.log()).sum()+(pa*pa.log()).sum(),True)
        self.assertTrue(torch.equal(cluster_representation(q,Config()),s))
        ga=torch.autograd.grad(contrast+entropy,(q,qa),retain_graph=True)
        gb=torch.autograd.grad(ref,(q,qa))
        for a,b in zip(ga,gb):self.assert_tensor(a,b,True)

    def test_full_objective_composition_matches_legacy(self):
        import torch
        from torch.nn import functional as F
        from loster2026.config import Config
        from loster2026.losses import objective
        x=torch.randn(4,7,1,dtype=torch.float64);xa=torch.randn_like(x)
        xr=torch.randn_like(x);xar=torch.randn_like(x)
        z=torch.randn(4,5,dtype=torch.float64);za=torch.randn_like(z)
        co=torch.randn(3,5,dtype=torch.float64);ca=torch.randn_like(co)
        q=F.one_hot(torch.tensor([0,1,2,1]),3).double();qa=q.flip(0)
        actual=objective(x,xa,xr,xar,z,za,q,qa,co,ca,Config())["total"]
        rec=F.mse_loss(xr,x)+F.mse_loss(xar,xa)
        km=.5*(self.loss["KMeansLoss"](co.numpy())(z,q)+self.loss["KMeansLoss"](ca.numpy())(za,qa))
        expected=rec+km+self.loss["InstanceContrastiveLoss"]()(z,za)+self.loss["ClusterContrastiveLoss"]()(q,qa)
        self.assert_tensor(actual,expected,True)

    def test_noresoftmax_changes_only_cluster_branch(self):
        import torch
        from torch.nn import functional as F
        from loster2026.config import Config,VARIANTS
        from loster2026.losses import objective,cluster_representation
        a=Config();b=replace(a,variant=VARIANTS[1])
        x=torch.randn(4,7,1);xa=torch.randn_like(x);xr=torch.randn_like(x);xar=torch.randn_like(x)
        z=torch.randn(4,5);za=torch.randn_like(z);co=torch.randn(5,5);ca=torch.randn_like(co)
        q=F.one_hot(torch.tensor([0,0,1,1]),5).float().requires_grad_()
        qa=F.one_hot(torch.tensor([1,1,0,0]),5).float().requires_grad_()
        first=objective(x,xa,xr,xar,z,za,q,qa,co,ca,a)
        second=objective(x,xa,xr,xar,z,za,q,qa,co,ca,b)
        for key in ("reconstruction","kmeans","instance"):
            self.assertTrue(torch.equal(first[key],second[key]))
        self.assertTrue(torch.equal(cluster_representation(q,b),q))
        self.assertFalse(torch.equal(cluster_representation(q,a),q))
        second["total"].backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertLess(q.grad.abs().max().item(),1e4)

    def test_alignedinit_loss_is_identical_to_legacy(self):
        import torch
        from torch.nn import functional as F
        from loster2026.config import Config,VARIANTS
        from loster2026.losses import cluster_terms
        q=F.one_hot(torch.tensor([0,1,1]),3).float();qa=q.flip(0)
        a=cluster_terms(q,qa,Config())
        b=cluster_terms(q,qa,replace(Config(),variant=VARIANTS[2]))
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))

    def test_complete_checkpoint_roundtrip_and_missing_active_rejection(self):
        import torch
        from loster2026.architecture import TwoViewModel
        from loster2026.assignments import deterministic_assignments
        from loster2026.config import Config
        from loster2026.checkpointing import save_checkpoint,load_checkpoint,trusted_load
        c=replace(Config(),latent_dim=8,dataset="Tiny")
        model=TwoViewModel(12,2,c);model.eval()
        x=torch.randn(6,12,1)
        with torch.no_grad():
            _,z=model.original(x)
            model.centers_original.copy_(z[:2])
            model.centers_augmented.copy_(z[:2]+17)
            before=deterministic_assignments(z,model.centers_original).clone()
        self.assertEqual(before[:2].tolist(),[0,1])
        augmented_centers=model.centers_augmented.detach().clone()
        with tempfile.TemporaryDirectory(prefix="loster2026-checkpoint-") as root:
            path=Path(root)/"Tiny"/c.variant/"seed-0"/"complete.pt"
            save_checkpoint(path,model,c,2,6.5,None,None,before.tolist(),{},{"git":{"sha":"fixture"}},None)
            with self.assertRaises(FileExistsError):save_checkpoint(path,model,c,2,6.5,None,None,[],{}, {},None)
            del model
            restored,payload=load_checkpoint(path)
            with torch.no_grad():
                _,z=restored.original(x)
                after=deterministic_assignments(z,restored.centers_original)
            self.assertTrue(torch.equal(before,after))
            self.assertTrue(torch.equal(augmented_centers,restored.centers_augmented))
            self.assertEqual(payload["final_temperature"],6.5)
            bad=trusted_load(path);bad["model_state"].pop("centers_original")
            broken=Path(root)/"broken.pt";torch.save(bad,broken)
            with self.assertRaisesRegex(ValueError,"ACTIVE"):load_checkpoint(broken)

    def test_device_and_parameter_instrumentation(self):
        from loster2026.reproducibility import resolve_device
        from loster2026.architecture import TwoViewModel
        from loster2026.efficiency import memory_and_parameters
        from loster2026.config import Config
        self.assertEqual(str(resolve_device("cpu")),"cpu")
        model=TwoViewModel(12,3,replace(Config(),latent_dim=8))
        info=memory_and_parameters(model,resolve_device("cpu"))
        self.assertEqual(info["objective_active_parameters"],sum(p.numel() for p in model.parameters()))
        self.assertIsNone(info["peak_gpu_allocated_bytes"])
        self.assertGreater(info["deployed_original_parameters"],0)

@unittest.skipUnless(available("numpy","scipy","torch"),"Torch/NumPy/SciPy missing/unimportable")
class PairedInitializationTests(unittest.TestCase):
    def test_alignedinit_only_permutes_augmented_centers_and_replays_rng(self):
        import numpy as np
        import torch
        from loster2026.architecture import TwoViewModel
        from loster2026.config import Config,VARIANTS
        from loster2026.reproducibility import seed_all,capture_rng
        from loster2026.training import initialize_variant,optimizer_and_scheduler
        c=replace(Config(),latent_dim=8)
        seed_all(23);source=TwoViewModel(12,3,c)
        with torch.no_grad():
            source.centers_original.copy_(torch.randn(3,8))
            source.centers_augmented.copy_(torch.randn(3,8))
        initial={"model_state":source.state_dict(),"labels_original":np.array([0,0,1,1,2,2]),
                 "labels_augmented":np.array([2,2,0,0,1,1]),"rng_joint_boundary":capture_rng()}
        a,_=initialize_variant(initial,12,3,c,torch.device("cpu"));draw_a=torch.rand(5)
        b,_=initialize_variant(initial,12,3,replace(c,variant=VARIANTS[1]),torch.device("cpu"));draw_b=torch.rand(5)
        aligned,info=initialize_variant(initial,12,3,replace(c,variant=VARIANTS[2]),torch.device("cpu"))
        draw_c=torch.rand(5)
        self.assertTrue(torch.equal(draw_a,draw_b));self.assertTrue(torch.equal(draw_a,draw_c))
        for key,value in a.state_dict().items():
            self.assertTrue(torch.equal(value,b.state_dict()[key]))
            if key!="centers_augmented":self.assertTrue(torch.equal(value,aligned.state_dict()[key]))
        self.assertTrue(torch.equal(aligned.centers_augmented,a.centers_augmented[[2,0,1]]))
        self.assertEqual(info["matching_calls"],1)
        for model,conf in [(a,c),(b,replace(c,variant=VARIANTS[1])),(aligned,replace(c,variant=VARIANTS[2]))]:
            opt,sch=optimizer_and_scheduler(model,conf)
            self.assertEqual(opt.defaults["lr"],.01)
            self.assertEqual(opt.defaults["momentum"],0.)
            self.assertEqual(sch.step_size,5)

@unittest.skipUnless(available("numpy","scipy","torch"),"Torch/NumPy/SciPy missing/unimportable")
class GradientAndPretrainTests(unittest.TestCase):
    def test_one_epoch_pretrain_active_path(self):
        import numpy as np
        import torch
        from loster2026.config import Config
        from loster2026.architecture import Autoencoder
        from loster2026.training import _loader, _pretrain
        from loster2026.data import normalize_rows
        c=replace(Config(),latent_dim=8,pretrain_epochs=1,batch_size=8,num_workers=0)
        x=normalize_rows(np.random.RandomState(3).normal(size=(16,24))).astype(np.float32)
        loader=_loader(x,-x,c)
        model=Autoencoder(24,c)
        before={k:v.clone() for k,v in model.state_dict().items()}
        losses=_pretrain(model,loader,0,c,torch.device("cpu"))
        self.assertEqual(len(losses),1)
        self.assertTrue(np.isfinite(losses).all())
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items()))

    def test_component_gradients_do_not_change_rng_grads_or_update(self):
        import torch
        from loster2026.config import Config,VARIANTS
        from loster2026.architecture import TwoViewModel
        from loster2026.assignments import rbf_logits,hard_assignments
        from loster2026.losses import objective
        from loster2026.diagnostics import component_gradients
        from loster2026.reproducibility import seed_all,capture_rng,restore_rng
        from loster2026.training import optimizer_and_scheduler
        x=torch.randn(4,12,1);xa=-x
        for variant in VARIANTS:
            c=replace(Config(),latent_dim=8,variant=variant,num_workers=0)
            seed_all(12);model=TwoViewModel(12,5,c)
            with torch.no_grad():
                model.centers_original.copy_(torch.randn(5,8))
                model.centers_augmented.copy_(torch.randn(5,8))
            state={k:v.clone() for k,v in model.state_dict().items()}
            boundary=capture_rng()
            updated=[]
            for diagnostic in (False,True):
                model.load_state_dict(state);restore_rng(boundary)
                opt,_=optimizer_and_scheduler(model,c);opt.zero_grad()
                xr,z=model.original(x);xar,za=model.augmented(xa)
                q=hard_assignments(rbf_logits(z,model.centers_original),2.)
                qa=hard_assignments(rbf_logits(za,model.centers_augmented),2.)
                terms=objective(x,xa,xr,xar,z,za,q,qa,model.centers_original,model.centers_augmented,c)
                if diagnostic:
                    before_rng=torch.get_rng_state().clone()
                    before_grads=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
                    norms=component_gradients(terms,model,c.alpha)
                    self.assertTrue(torch.equal(before_rng,torch.get_rng_state()))
                    for parameter, before in zip(model.parameters(),before_grads):
                        if before is None:
                            self.assertIsNone(parameter.grad)
                        else:
                            self.assertTrue(torch.equal(parameter.grad,before))
                    self.assertTrue(all(value>=0 for groups in norms.values() for value in groups.values()))
                terms["total"].backward();opt.step()
                updated.append({k:v.clone() for k,v in model.state_dict().items()})
            for key in updated[0]:
                self.assertTrue(torch.equal(updated[0][key],updated[1][key]),key)

@unittest.skipUnless(available("numpy","scipy","sklearn","torch"),"Complete scientific stack unavailable: synthetic end-to-end BLOCKED")
class EndToEndTests(unittest.TestCase):
    def test_three_variants_synthetic_end_to_end_and_checkpoint(self):
        from loster2026.smoke import synthetic_smoke
        from loster2026.config import VARIANTS
        results=synthetic_smoke()
        self.assertEqual([r["variant"] for r in results],list(VARIANTS))
        self.assertTrue(all(r["complete"] and r["finite"] for r in results))
        self.assertTrue(all(1<=r["final_epoch"]<=2 for r in results))
        self.assertEqual(len({r["paired_cache_key"] for r in results}),1)
