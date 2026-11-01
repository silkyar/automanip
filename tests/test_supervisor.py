import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from automanip.supervisor import StopPolicy, RunpodClient, load_api_key


class SupervisorTests(unittest.TestCase):
    def test_deadline_is_independent_of_training_status(self):
        p=StopPolicy(100,600,5,.6)
        self.assertIsNone(p.reason(699))
        self.assertEqual(p.reason(700),"hard_deadline")
        self.assertEqual(p.reason(700,True),"hard_deadline")

    def test_cost_envelope_and_completion(self):
        p=StopPolicy(100,10000,.1,.6)
        self.assertEqual(p.reason(700),"cost_envelope")
        self.assertEqual(p.reason(101,True),"batch_finished")
        self.assertIsNone(p.reason(99))

    def test_env_is_data_not_shell_code(self):
        with tempfile.TemporaryDirectory() as temp:
            f=Path(temp)/".env"
            f.write_text("OTHER=value\nRUNPOD_API_KEY='example-key'\n")
            self.assertEqual(load_api_key(f),"example-key")
            f.write_text("RUNPOD_API_KEY=a\nRUNPOD_API_KEY=b\n")
            with self.assertRaises(ValueError):load_api_key(f)

    def test_requests_only_touch_explicit_pod_and_stop_action(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def read(self,*a):return b'{"id":"ours","status":"EXITED"}'
        with patch("urllib.request.urlopen",return_value=Response()) as mocked:
            RunpodClient("secret","ours").stop()
            request=mocked.call_args.args[0]
            self.assertEqual(request.full_url,"https://api.runpod.io/v2/pods/ours/action")
            self.assertEqual(json.loads(request.data),{"action":"stop"})
        with self.assertRaises(ValueError):RunpodClient("secret","../foreign")

    def test_diagnostics_completion_waits_but_training_failure_stops(self):
        from scripts.supervise_pod import completion_probe_command
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            command=completion_probe_command(temp,"diagnostics.exit")
            completed=lambda: subprocess.run(["sh","-c",command],capture_output=True).returncode==0
            (root/"setup.exit").write_text("0\n")
            (root/"batch.exit").write_text("0\n")
            self.assertFalse(completed())
            (root/"diagnostics.exit").write_text("0\n")
            self.assertTrue(completed())
            (root/"diagnostics.exit").unlink()
            (root/"batch.exit").write_text("1\n")
            self.assertTrue(completed())
        with self.assertRaises(ValueError):completion_probe_command("/workspace/ours","../foreign")


if __name__=="__main__":unittest.main()
