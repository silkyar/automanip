import ast
import math
import importlib.util
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CAPG checks require optional PyTorch')
class CAPGTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch

    def test_boundary_mass_matches_integrated_gaussian_and_ignores_latent_tail(self):
        t=self.torch
        from automanip.capg import clipped_gaussian_log_prob
        p=t.distributions.Normal(t.tensor(.2,dtype=t.float64),t.tensor(.7,dtype=t.float64))
        values=clipped_gaussian_log_prob(p,t.tensor([-10.,-1.,0.,1.,10.],dtype=t.float64))
        self.assertEqual(values[0].item(),values[1].item())
        self.assertEqual(values[3].item(),values[4].item())
        self.assertAlmostEqual(values[2].item(),p.log_prob(t.tensor(0.)).item(),places=12)
        lower=.5*(1+math.erf((-1-.2)/(.7*math.sqrt(2))))
        upper=.5*math.erfc((1-.2)/(.7*math.sqrt(2)))
        self.assertAlmostEqual(values[0].exp().item(),lower,places=12)
        self.assertAlmostEqual(values[-1].exp().item(),upper,places=12)

    def test_extreme_boundary_scores_and_gradients_are_finite(self):
        t=self.torch
        from automanip.capg import clipped_gaussian_log_prob
        for dtype in (t.float32,t.float64):
            means=t.tensor([-20.,0.,20.],dtype=dtype,requires_grad=True)
            scales=t.tensor([.05,.8,.05],dtype=dtype,requires_grad=True)
            p=t.distributions.Normal(means,scales)
            values=clipped_gaussian_log_prob(p,t.tensor([10.,0.,-10.],dtype=dtype))
            values.sum().backward()
            self.assertTrue(t.isfinite(values).all())
            self.assertTrue(t.isfinite(means.grad).all());self.assertTrue(t.isfinite(scales.grad).all())

    def test_toy_bounded_control_gradient_is_unbiased_with_lower_variance(self):
        t=self.torch
        from automanip.capg import clipped_gaussian_log_prob
        generator=t.Generator().manual_seed(3821);mu=.2;sigma=1.5;n=100_000
        actions=t.randn(n,generator=generator,dtype=t.float64)*sigma+mu
        means=t.full((n,),mu,dtype=t.float64,requires_grad=True)
        p=t.distributions.Normal(means,t.full((n,),sigma,dtype=t.float64))
        scores=t.autograd.grad(clipped_gaussian_log_prob(p,actions).sum(),means)[0]
        reward=actions.clamp(-1.,1.)
        capg=reward*scores;ordinary=reward*(actions-mu)/(sigma*sigma)
        exact=.5*(math.erf((1-mu)/(sigma*math.sqrt(2)))-math.erf((-1-mu)/(sigma*math.sqrt(2))))
        self.assertAlmostEqual(capg.mean().item(),exact,delta=.006)
        self.assertAlmostEqual(ordinary.mean().item(),exact,delta=.006)
        self.assertLess(capg.var().item(),ordinary.var().item()*.9)

    def test_derivative_changes_only_likelihood_on_top_of_common_instrumentation(self):
        from automanip.capg import build_trainer
        official=(Path(__file__).resolve().parents[1]/'third_party/maniskill/ppo.py').read_bytes()
        control=build_trainer(official,'full-horizon-control').decode()
        variant=build_trainer(official,'full-horizon-capg').decode();ast.parse(variant)
        before,after=variant.split('# BEGIN CAPG FUNCTION\n',1)
        _,after=after.split('# END CAPG FUNCTION\n',1)
        stripped=(before+after).replace('clipped_gaussian_log_prob(probs, action).sum(1)','probs.log_prob(action).sum(1)')
        self.assertEqual(stripped,control)
        with self.assertRaises(ValueError):build_trainer(official,'arbitrary-code')


if __name__=='__main__':unittest.main()
