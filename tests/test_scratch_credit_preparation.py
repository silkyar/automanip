import ast
from pathlib import Path
from types import SimpleNamespace
import textwrap
import unittest

import torch

from scripts.prepare_scratch_credit import BASE, MC_BLOCK, build_sources, gae_region


class ScratchCreditPreparationTests(unittest.TestCase):
    def test_two_programs_differ_only_in_target_block(self):
        programs = build_sources(BASE.read_bytes())
        left, right = [programs[k] for k in ['ppo-gae','ppo-finite-mc']]
        a,b = gae_region(left);c,d = gae_region(right)
        self.assertEqual(left[:a]+left[b:], right[:c]+right[d:])
        original = BASE.read_text();e,f = gae_region(original)
        self.assertEqual(left[a:b], original[e:f])
        for program in [left,right]:
            old_class=next(n for n in ast.parse(original).body if isinstance(n,ast.ClassDef) and n.name=='Agent')
            new_class=next(n for n in ast.parse(program).body if isinstance(n,ast.ClassDef) and n.name=='Agent')
            self.assertEqual(ast.dump(old_class),ast.dump(new_class))

    def test_mc_terminal_value_and_rewards(self):
        rewards=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
        values=torch.full_like(rewards,1000.)
        namespace=dict(torch=torch,args=SimpleNamespace(num_steps=3,gamma=.9),rewards=rewards,values=values,
                       dones=torch.zeros_like(rewards),next_done=torch.ones(2))
        exec(compile(textwrap.dedent(MC_BLOCK),'finite-mc-test','exec'),namespace)
        expected=torch.tensor([[1+2.7+4.05,2+3.6+4.86],[3+4.5,4+5.4],[5,6]])
        torch.testing.assert_close(namespace['returns'],expected)
        torch.testing.assert_close(namespace['advantages'],expected-values)

    def test_incomplete_episodes_cannot_silently_become_mc(self):
        for interior,last in [(True,True),(False,False)]:
            dones=torch.zeros(3,2);dones[1,0]=int(interior)
            namespace=dict(torch=torch,args=SimpleNamespace(num_steps=3,gamma=.9),rewards=torch.ones(3,2),
                           values=torch.zeros(3,2),dones=dones,next_done=torch.full((2,),int(last)))
            with self.assertRaisesRegex(ValueError,'complete aligned'):
                exec(compile(textwrap.dedent(MC_BLOCK),'finite-mc-test','exec'),namespace)


if __name__=='__main__':unittest.main()
