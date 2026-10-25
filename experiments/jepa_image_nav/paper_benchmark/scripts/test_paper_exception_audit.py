"""CPU-only accounting regression; fixtures cannot create navigation scores."""
import copy
import ast
import json
import tempfile
import unittest
from pathlib import Path
from paper_freeze import sha
from paper_exception_audit import classify, validate_exception
from paper_exception_replay import replay_kernel
from paper_freeze import ROOT


class AccountingTests(unittest.TestCase):
    def test_replay_kernel_detects_changed_physics_or_score(self):
        original=(ROOT/'tools/paper_replay.py').read_text()
        candidate=(ROOT/'tools/paper_exception_replay.py').read_text()
        baseline=replay_kernel(ast.parse(original))
        self.assertEqual(baseline,replay_kernel(ast.parse(candidate)))
        for a,b in [('-.25','-.24'), ('fraction >= .025','fraction >= .02'),
                    ('< 1e-4','< 1e-3'), ('(.5 if instance else 1.)','(.6 if instance else 1.)')]:
            self.assertNotEqual(baseline,replay_kernel(ast.parse(candidate.replace(a,b))))

    def test_partition(self):
        self.assertEqual(classify(True,False), 'valid')
        self.assertEqual(classify(False,True), 'exception')
        self.assertEqual(classify(False,False), 'pending')
        with self.assertRaises(AssertionError):
            classify(True,True)

    def test_narrow_exception_and_immutability(self):
        with tempfile.TemporaryDirectory(prefix='unigoal-audit-fixture-') as tmp:
            base=Path(tmp); folder=base/'episode'; folder.mkdir()
            episode=dict(episode_id='test-id', scene_id='test-scene')
            runtime=base/'runtime.json'; runtime.write_text('{}')
            (folder/'panel.jsonl').write_text(json.dumps(episode)+'\n')
            failure=dict(error='IndexError', detail='index 767 is out of bounds for axis 1 with size 722')
            (folder/'runtime_failure.json').write_text(json.dumps(failure))
            log=base/'native.log'; log.write_text(
                'src/graph/graph.py in get_goal\nsrc/utils/fmm/fmm_planner.py in set_goal\n'
                'self.traversible[goal_x, goal_y]\n')
            c=dict(episode_id='test-id',scene_id='test-scene',
                status='native_implementation_exception_uncompleted', performance_score_valid=False,
                counted_as_navigation_failure=False, original_native_framework_preserved=True,
                no_synthetic_STOP_or_score=True, frozen_runtime_sha256=sha(runtime),
                episode_panel_sha256=sha(folder/'panel.jsonl'),exception=failure,
                evidence_log_sha256=sha(log),protected_attempt_file_hashes={
                    str(p):sha(p) for p in folder.iterdir()})
            validate_exception(c,episode,folder,log,runtime)
            for field, bad in [('counted_as_navigation_failure',True),('performance_score_valid',True),
                               ('episode_id','substitution'),('frozen_runtime_sha256','changed')]:
                altered=copy.deepcopy(c);altered[field]=bad
                with self.assertRaises(AssertionError):
                    validate_exception(altered,episode,folder,log,runtime)
            (folder/'added_score.json').write_text('{}')
            with self.assertRaises(AssertionError):
                validate_exception(c,episode,folder,log,runtime)
            (folder/'added_score.json').unlink()
            log.write_text('Different engine failure')
            with self.assertRaises(AssertionError):
                validate_exception(c,episode,folder,log,runtime)


if __name__=='__main__':
    unittest.main()
