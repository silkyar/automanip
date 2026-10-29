import copy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from automanip.research import validate_campaign, training_command, snapshot_file, paired_development, CheckpointMirror


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/"configs/stackcube-credit-20261026.json").read_text())

    def test_only_preregistered_parameters_are_accepted(self):
        validate_campaign(self.config)
        for key, value in (("training_seeds",[3,2,1]),("requested_transitions",50_000_000),("evaluation_split","confirmation")):
            candidate=copy.deepcopy(self.config);candidate[key]=value
            with self.assertRaises(ValueError):validate_campaign(candidate)
        candidate=copy.deepcopy(self.config);candidate["candidates"][0]["reward_scale"]=2
        with self.assertRaises(ValueError):validate_campaign(candidate)

    def test_training_change_is_only_gamma(self):
        a=training_command("python","ppo.py",self.config["candidates"][0],1)
        b=training_command("python","ppo.py",self.config["candidates"][1],1)
        self.assertEqual([(i,x,y) for i,(x,y) in enumerate(zip(a,b)) if x!=y],
                         [(a.index("gamma090-seed-1"),"gamma090-seed-1","gamma095-seed-1"),(len(a)-1,"0.9","0.95")])
        self.assertNotIn("--checkpoint",a)

    def test_snapshots_are_bounded_and_reject_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/"source").write_bytes(b"abc")
            snapshot_file(p/"source",p/"copy",3)
            self.assertEqual((p/"copy").read_bytes(),b"abc")
            self.assertFalse((p/"copy").stat().st_mode & 0o222)
            with self.assertRaises(ValueError):snapshot_file(p/"source",p/"large",2)
            (p/"link").symlink_to(p/"source")
            with self.assertRaises(OSError):snapshot_file(p/"link",p/"linked",3)

    def test_paired_comparison_never_accepts_confirmation(self):
        def report(successes):
            return dict(split="development",smoke=False,episode_count=128,failed_episodes=0,protocol_sha256="same",
                        episodes=[dict(outcome=dict(terminal_success=i in successes,failed=False,terminated=False,truncated=True)) for i in range(128)])
        a,b=report({1,2}),report({2,3})
        r=paired_development(a,b)
        self.assertEqual((r["paired_wins"],r["paired_losses"],r["success_rate_delta"]),(1,1,0))
        a["split"]="confirmation"
        with self.assertRaises(ValueError):paired_development(a,b)

    def test_checkpoint_mirror_waits_for_stable_files_and_flushes_final(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/"source";source.mkdir()
            (source/"ckpt_1.pt").write_bytes(b"checkpoint")
            (source/"arbitrary.py").write_text("untrusted")
            mirror=CheckpointMirror(source,root/"durable")
            mirror.poll();self.assertFalse((root/"durable/ckpt_1.pt").exists())
            mirror.poll();self.assertEqual((root/"durable/ckpt_1.pt").read_bytes(),b"checkpoint")
            self.assertFalse((root/"durable/arbitrary.py").exists())
            with mirror:
                (source/"final_ckpt.pt").write_bytes(b"final")
            self.assertEqual((root/"durable/final_ckpt.pt").read_bytes(),b"final")
            manifest=json.loads((root/"durable/durable-files.json").read_text())
            self.assertEqual(manifest["final_ckpt.pt"]["bytes"],5)

    @unittest.skipUnless(sys.platform=="linux","Linux Landlock enforcement test")
    def test_landlock_denies_even_world_readable_unlisted_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);allow=root/"allowed";allow.mkdir();deny=root/"denied";deny.mkdir()
            (allow/"data").write_text("allowed");(deny/"data").write_text("denied")
            policy=root/"policy.json"
            policy.write_text(json.dumps(dict(read_execute=['/usr','/lib','/lib64','/bin','/etc',str(Path(sys.executable).parents[1]),str(allow)],read_write=[])))
            code=f"from pathlib import Path\nassert Path({str(allow/'data')!r}).read_text()=='allowed'\ntry: Path({str(deny/'data')!r}).read_text()\nexcept PermissionError: pass\nelse: raise AssertionError('Landlock did not deny access')"
            script=Path(__file__).resolve().parents[1]/"scripts/sandbox_exec.py"
            subprocess.run([sys.executable,str(script),'--policy',str(policy),'--',sys.executable,'-c',code],check=True,cwd='/tmp')


if __name__=="__main__":unittest.main()
