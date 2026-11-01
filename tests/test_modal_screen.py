import importlib.util
from pathlib import Path
import os
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('screen_contract',Path(__file__).resolve().parents[1]/'deploy/modal/screen_contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class ScreenContracts(unittest.TestCase):
    def test_exact_two_methods_seed_one_and_fixed_source(self):
        for method in c.REVIEWED:
            item=c.job(method);args=c.command('python','source.py',item)
            self.assertEqual(item['seed'],1);self.assertEqual(args[args.index('--num-eval-envs')+1],'8')
            self.assertNotIn('--checkpoint',args);self.assertIn('--no-partial-reset',args)
            changed=dict(item,seed=2)
            with self.assertRaises(ValueError):c.command('python','source.py',changed)
        with self.assertRaises(ValueError):c.job('new-method')

    def test_original_deadline_never_refreshes_after_queue_or_retry(self):
        remaining,deadline=c.remaining_lease(2000.,1000.,5.)
        self.assertEqual(remaining,1000.);self.assertEqual(deadline,985.)
        self.assertEqual(c.remaining_lease(2000.,1500.,50.),(500.,530.))
        with self.assertRaises(RuntimeError):c.remaining_lease(2000.,1980.,0.)

    def test_checkpoint_cadence_and_censored_points(self):
        files={'ckpt_101.pt','ckpt_201.pt','ckpt_301.pt','ckpt_401.pt','final_ckpt.pt'}
        self.assertEqual([x['steps'] for x in c.select_checkpoints(files,c.MAX_STEPS)],
                         [5_120_000,10_240_000,15_360_000,20_480_000,c.MAX_STEPS])
        self.assertEqual(c.select_checkpoints({'final_ckpt.pt'},5_120_000),
                         [{'steps':5_120_000,'filename':'final_ckpt.pt','point':'prespecified-final-coincidence'}])
        with self.assertRaises(ValueError):c.select_checkpoints({'final_ckpt.pt'},10_240_000)
        with self.assertRaises(ValueError):c.select_checkpoints(files,5_120_000)
        with self.assertRaises(ValueError):c.select_checkpoints({'ckpt_426.pt','final_ckpt.pt'},20_480_000)
        coincident=c.select_checkpoints({'ckpt_101.pt','final_ckpt.pt'},5_120_000)
        self.assertEqual(len(coincident),1);self.assertEqual(coincident[0]['filename'],'final_ckpt.pt')

    def test_artifact_reads_refuse_symlinks_fifo_oversize_and_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'ok').write_bytes(b'1234')
            self.assertEqual(c.read_regular(root,'ok',4),b'1234')
            with self.assertRaises(ValueError):c.read_regular(root,'ok',3)
            with self.assertRaises(ValueError):c.read_regular(root,'../ok',10)
            (root/'link').symlink_to(root/'ok')
            with self.assertRaises(OSError):c.read_regular(root,'link',10)
            (root/'dir').mkdir();(root/'dir/file').write_text('x');(root/'dirlink').symlink_to(root/'dir')
            with self.assertRaises(OSError):c.read_regular(root,'dirlink/file',10)
            os.mkfifo(root/'fifo')
            with self.assertRaises(ValueError):c.read_regular(root,'fifo',10)

    def test_resource_ceiling_has_reserve_under_three_dollars(self):
        self.assertLess(c.conservative_cost()['subtotal_usd'],2.8)
        self.assertGreater(c.conservative_cost()['reserve_usd'],.2)


if __name__=='__main__':unittest.main()
