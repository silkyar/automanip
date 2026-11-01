import ast
from pathlib import Path
import unittest

from scripts.prepare_gripper_hold import sources
from scripts.prepare_scratch_credit import BASE, gae_region, update_region


class HoldPreparationTests(unittest.TestCase):
    def test_gaussian_agent_critic_and_ppo_objective_unchanged(self):
        original = BASE.read_text()
        original_class = next(n for n in ast.parse(original).body if isinstance(n, ast.ClassDef) and n.name == 'Agent')
        for name, program in sources().items():
            actual_class = next(n for n in ast.parse(program).body if isinstance(n, ast.ClassDef) and n.name == 'Agent')
            self.assertEqual(ast.dump(original_class), ast.dump(actual_class))
            a, b = gae_region(program)
            c, d = gae_region(original)
            self.assertEqual(program[a:b], original[c:d])
            actual_update = update_region(program)
            for event in ('reserved', 'completed'):
                actual_update = actual_update.replace(f"                automanip_event('optimizer_{event}',automanip_optimizer_steps+1)\n", '')
            self.assertEqual(actual_update, update_region(original))
            self.assertNotIn('torch.load(', program.replace('agent.load_state_dict(torch.load(args.checkpoint))', ''))

    def test_masked_programs_only_differ_in_prior_switch(self):
        programs = sources()
        self.assertEqual(programs['masked'].replace('configure(agent.actor_mean, False)',
                                                  'configure(agent.actor_mean, True)'), programs['hold'])
        self.assertNotIn('configure(agent.actor_mean', programs['full'])


if __name__ == '__main__':
    unittest.main()
