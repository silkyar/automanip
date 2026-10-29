import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('modal_bundle',ROOT/'deploy/modal/bundle.py')
bundle=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundle)
spec=importlib.util.spec_from_file_location('modal_transport',ROOT/'deploy/modal/transport.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)


class ModalBundleTests(unittest.TestCase):
    def test_trainer_image_excludes_benchmark_evaluator_and_credentials(self):
        self.assertEqual(bundle.source_files('trainer'),('third_party/maniskill/ppo.py',))
        with tempfile.TemporaryDirectory() as temporary:
            stage=bundle.build_context('trainer',Path(temporary)/'stage')
            files={str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}
            self.assertEqual(files,{'third_party/maniskill/ppo.py','source-manifest.json'})

    def test_validation_bundle_contains_every_frozen_file_and_no_extra_secrets(self):
        protocol=json.loads((ROOT/'benchmarks/stackcube-v1/protocol.json').read_text())
        files=set(bundle.source_files('validation'))
        self.assertTrue(set(protocol['files']).issubset(files))
        self.assertFalse(any(name.startswith(('.runtime/','.env','.git/')) for name in files))
        self.assertNotIn('src/automanip/supervisor.py',files)
        self.assertNotIn('scripts/run_settling_campaign.py',files)

    def test_only_known_image_roles_are_accepted(self):
        with self.assertRaises(ValueError):bundle.source_files('../credentials')

    def test_restricted_transport_is_bounded_and_preserves_exact_bytes(self):
        raw=b'{"weights":[1.25,2.0]}'
        packed=transport.pack_weights(raw)
        self.assertEqual(transport.unpack_weights(packed),raw)
        with self.assertRaises(ValueError):transport.unpack_weights(packed,max_bytes=3)
        with self.assertRaises(ValueError):transport.unpack_weights(packed+packed)
        with self.assertRaises(ValueError):transport.unpack_weights(packed[:-2])


if __name__=='__main__':unittest.main()
