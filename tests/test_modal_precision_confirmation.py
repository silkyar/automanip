import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('precision_confirmation_contract',ROOT/'deploy/modal/precision_confirmation_contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class ConfirmationAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((ROOT/'deploy/modal/precision-confirmation-plan.json').read_bytes())
        self.plan.update(launch_enabled=True,root_reviewed=True)

    def test_fixed_scope_and_default_denial(self):
        self.assertEqual(c.validate(self.plan),c.DIGESTS)
        for key in ('launch_enabled','root_reviewed'):
            plan=copy.deepcopy(self.plan);plan[key]=False
            with self.assertRaises(PermissionError):c.validate(plan)

    def test_reject_policy_swaps_and_query_budget_changes(self):
        for key,value in dict(confirmation_weights_sha256=list(reversed(c.DIGESTS)),
                              training_seeds=[1,2,4],split='development',episodes_per_policy=1000,
                              max_parallel=4,evaluation_timeout_seconds=1100,budget_cap_usd=2,
                              retries=1,new_training_runs=1,acceptance_rule={}).items():
            plan=copy.deepcopy(self.plan);plan[key]=value
            with self.assertRaises(ValueError,msg=key):c.validate(plan)

    def test_registered_scorer_uses_exact_frozen_function_body(self):
        tree=ast.parse((ROOT/'deploy/modal/precision_confirmation.py').read_bytes())
        extraction=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='frozen_confirmation_body')
        namespace={'ast':ast,'__name__':__name__}
        exec(compile(ast.Module(body=[extraction],type_ignores=[]),'trusted-body-test','exec'),namespace)
        source=(ROOT/'deploy/modal/evaluate.py').read_bytes()
        original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='frozen_evaluate')
        recovered=namespace['frozen_confirmation_body'](source)
        # A separate extraction, rather than candidate or replacement source,
        # must generate byte-identical evaluator bytecode and constants.
        original.decorator_list=[]
        expected={'__name__':__name__}
        exec(compile(ast.Module(body=[original],type_ignores=[]),'expected-frozen','exec'),expected)
        self.assertEqual(recovered.__code__.co_code,expected['frozen_evaluate'].__code__.co_code)
        self.assertEqual(recovered.__code__.co_consts,expected['frozen_evaluate'].__code__.co_consts)


if __name__=='__main__':unittest.main()
