import ast
import unittest
from scripts.prepare_feature_value import build_programs
from scripts.prepare_scratch_credit import BASE


class FeatureValuePreparationTests(unittest.TestCase):
    def test_methods_differ_only_in_value_clock(self):
        programs=build_programs();state=programs['ppo-feature-value'];clock=programs['ppo-clock-feature-value']
        self.assertEqual(state.replace("args.num_envs,'state'","args.num_envs,'clock'"),clock)

    def test_agent_initialization_unchanged(self):
        def init(program):
            agent=next(n for n in ast.parse(program).body if isinstance(n,ast.ClassDef) and n.name=='Agent')
            return ast.dump(next(n for n in agent.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'))
        for program in build_programs().values():self.assertEqual(init(program),init(BASE.read_text()))

    def test_disabled_critic_is_not_accidentally_evaluated(self):
        for program in build_programs().values():
            agent=next(n for n in ast.parse(program).body if isinstance(n,ast.ClassDef) and n.name=='Agent')
            for method in agent.body:
                if isinstance(method,ast.FunctionDef) and method.name in ['get_value','get_action_and_value']:
                    self.assertNotIn('self.critic(',ast.unparse(method))


if __name__=='__main__':unittest.main()
