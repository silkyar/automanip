import ast,copy,hashlib,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'deploy/modal'))
import env_count_screen_contract as c
spec=importlib.util.spec_from_file_location('kl_tests',ROOT/'tests/test_modal_kl_value_screen.py');fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
class EnvCountContracts(unittest.TestCase):
    def fixture(self,method):
        scalars,rows=fixtures.fixture(c.SOURCE_METHOD);batch=c.batch_steps(method)
        for tag,values in scalars.items():
            for i,row in enumerate(values):
                row['steps']=(i+1)*batch
                if tag in ('time/step','audit/kl_value/new_environment_transitions'):row['value']=(i+1)*batch
        for i,row in enumerate(rows):row['new_environment_transitions']=(i+1)*batch
        return scalars,rows
    def test_exact_shared_source_and_only_env_cli_change(self):
        source=ROOT/'research/discovery/from-scratch-kl-value-v2/scratch-kl-stop-control.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),c.SOURCE_SHA)
        args=[]
        for m in c.ARMS:
            cmd=c.command('python','source.py',c.job(m));self.assertNotIn('--checkpoint',cmd)
            self.assertEqual(cmd[cmd.index('--num-envs')+1],str(c.ENV_COUNTS[m]));args.append(cmd)
        for args0,args1 in zip(*args):
            if args0!=args1:self.assertTrue(args0 in ('512',c.job(c.ARMS[0])['job_id']) and args1 in ('256',c.job(c.ARMS[1])['job_id']))
        self.assertNotEqual(c.job(c.ARMS[0])['job_id'],c.job(c.ARMS[1])['job_id'])
    def test_dynamic_checkpoints_and_censoring(self):
        for m,iteration in zip(c.ARMS,(201,401)):
            self.assertEqual(c.selected(m),{iteration:5120000})
            self.assertEqual([p['steps'] for p in c.select_checkpoints({'ckpt_'+str(iteration)+'.pt','final_ckpt.pt'},10240000,m)],[5120000,10240000])
            self.assertEqual(len(c.select_checkpoints({'final_ckpt.pt'},5120000,m)),1)
            with self.assertRaises(ValueError):c.select_checkpoints({'final_ckpt.pt'},10240000,m)
    def test_dynamic_audit_role_and_scalar_fail_closed(self):
        for m in c.ARMS:
            s,r=self.fixture(m);out=c.audit_accounting(s,m,2,fixtures.logs(r))
            self.assertEqual(out['optimizer_step_count'],276);self.assertEqual(out['measured_critic_only_optimizer_calls'],0)
            bad=copy.deepcopy(s);bad['audit/optimizer_steps'][0]['steps']=51200
            with self.assertRaises(ValueError):c.audit_accounting(bad,m,2,fixtures.logs(r))
            bad=copy.deepcopy(r);bad[0]['method']=m
            with self.assertRaises(ValueError):c.audit_accounting(s,m,2,fixtures.logs(bad))
            bad=copy.deepcopy(r);bad[0]['critic_only_optimizer_calls']=1
            with self.assertRaises(ValueError):c.audit_accounting(s,m,2,fixtures.logs(bad))
    def test_global_reuse_validates_before_spawn(self):
        d='a'*64;report=dict(protocol_sha256=c.PROTOCOL,weights_sha256=d,split='development',episode_count=128,failed_episodes=0,smoke=False)
        durable=dict(state='complete',protocol_sha256=c.PROTOCOL,weights_sha256=d,split='development',report=report,evaluation_seconds=1,gpu_actual_name='NVIDIA L40S',queue_seconds=0)
        self.assertTrue(c.existing_report(durable,d)['reused_global_report']);self.assertIsNone(c.existing_report(None,d))
        with self.assertRaises(ValueError):c.existing_report(dict(durable,state='failed'),d)
        tree=ast.parse((ROOT/'deploy/modal/env_count_screen.py').read_text())
        coordinator=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='coordinate_env_count')
        text=ast.unparse(coordinator)
        self.assertLess(text.index('existing_report(durable, digest)'),text.index('evaluator_function.spawn'))
        self.assertIn("key = item['job_id'] + '-' + digest",text)
    def test_default_denial_reference_lease_and_image_closure(self):
        p=json.loads((ROOT/'deploy/modal/env-count-screen-plan.json').read_bytes())
        with self.assertRaises(PermissionError):c.validate_admission(dict(p,launch_enabled=False,paid_launch_authorized=False,root_review=None))
        c.validate_admission(dict(p,launch_enabled=True,paid_launch_authorized=True,root_review={'test':True}))
        self.assertEqual(hashlib.sha256((ROOT/'.runtime/modal-env-count-reference-binding.json').read_bytes()).hexdigest(),c.REFERENCE_BINDING_SHA)
        with self.assertRaises(RuntimeError):c.remaining_lease(100,81,0)
        tree=ast.parse((ROOT/'deploy/modal/env_count_screen.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HELPER_FILES' for t in n.targets))
        with tempfile.TemporaryDirectory() as td:
            for name in ast.literal_eval(node.value):(Path(td)/name).write_bytes((ROOT/'deploy/modal'/name).read_bytes())
            r=subprocess.run([sys.executable,'-I','-S','-c','import sys;sys.path.insert(0,sys.argv[1]);import env_count_screen_contract',td],cwd=td,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
if __name__=='__main__':unittest.main()
