import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analyze_executed_rehearsal import analyze
from tests.test_readout_score_analysis import fixture


def prepare(root, improve):
    fixture(root)
    for path in root.rglob('report.json'):
        r=json.loads(path.read_text())
        for e in r['episodes']:
            e['diagnostics']=dict(ever={k:True for k in ('grasp','lift','transport','placement','release','stable_stack_10_steps')})
        if improve:
            successes={'base':0,'search':16,'unbranched':32}[r['method']]
            for i,e in enumerate(r['episodes']):e['outcome']['terminal_success']=i%64<successes
            r['successes']=successes*8;r['batches']=[dict(successes=successes) for _ in range(8)]
        path.write_text(json.dumps(r))


class ExecutedAnalysisTests(unittest.TestCase):
    def test_both_closed_form_and_refinement_controls_must_be_beaten(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);prepare(root,False)
            r=analyze(root)
            self.assertGreater(r['executed_minus_closed_form']['mean_gain'],.1)
            self.assertLess(r['executed_minus_raw_refinement']['mean_gain'],0)
            self.assertFalse(r['primary_screen_passed'])

    def test_matching_positive_contrasts_and_retention_pass(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);prepare(root,True)
            r=analyze(root)
            self.assertEqual(r['new_public_transitions'],153600)
            self.assertTrue(r['primary_screen_passed'])


if __name__=='__main__':unittest.main()
