import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('sac_screen_contract',Path(__file__).resolve().parents[1]/'deploy/modal/sac_screen_contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def report(actual=500000):
    entries=[{'path':'/trainer-output/run/actor-step-'+str(s),'transitions':s,'weights_sha256':str(s)} for s in [0,51200,102400,153600,204800,256000] if s<=actual]
    entries.append({'path':'/trainer-output/run/actor-final','transitions':actual,'weights_sha256':str(actual)})
    return {'transitions':actual,'exports':entries}


class SACScreenContractTests(unittest.TestCase):
    def test_exact_reviewed_cli(self):
        args=c.command('python','source','run')
        for key,value in c.CONFIG.items():self.assertEqual(args[args.index('--'+key.replace('_','-'))+1],str(value))
        self.assertNotIn('--checkpoint',args);self.assertEqual(c.CONFIG['seed'],1)

    def test_prespecified_points_and_final(self):
        self.assertEqual([e['steps'] for e in c.select_exports(report())],[51200,102400,256000,500000])
        self.assertEqual([e['steps'] for e in c.select_exports(report(256000))],[51200,102400,256000])
        self.assertEqual([e['steps'] for e in c.select_exports(report(32000))],[32000])
        bad=report();bad['exports']=[e for e in bad['exports'] if e['transitions']!=51200]
        with self.assertRaises(ValueError):c.select_exports(bad)
        bad=report(32000);bad['exports'].append({'path':'/actor-step-51200','transitions':51200,'weights_sha256':'x'})
        with self.assertRaises(ValueError):c.select_exports(bad)

    def test_bounded_budget(self):
        self.assertLess(c.cost_bound()['subtotal_usd'],1.4)
        self.assertGreater(c.cost_bound()['reserve_usd'],.1)


if __name__=='__main__':unittest.main()
