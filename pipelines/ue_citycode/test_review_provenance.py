"""Regressions for a scene review saved outside its source proposal directory."""
import json
from pathlib import Path
import tempfile
import unittest

from prepare_elevated_repair import digest_file, inherited_review_decisions


class ReviewProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'data/props.json'
        self.current = self.root / 'build/review/props.json'
        self.review = {'decisions_file': 'decisions.json', 'decisions': [{'action': 'translate'}]}
        self.write(self.source, {'agentic_review': self.review})
        self.write(self.source.parent / 'decisions.json', {'source': 'original'})
        # A same-name file next to the copied proposal must not win resolution.
        self.write(self.current.parent / 'decisions.json', {'source': 'unrelated'})
        self.value = {'agentic_review': self.review, 'scene_review_candidate': {
            'decisions_file': 'scene_decisions.json', 'baseline_proposal_sha256': digest_file(self.source)}}
        self.write(self.current, self.value)
        self.report = {'inputs': {'proposal': {'path': 'data/props.json', 'sha256': digest_file(self.source)}},
                       'outputs': {'proposal': {'sha256': digest_file(self.current)}}}
        self.report_path = self.current.parent / 'scene_decisions.json'
        self.write(self.report_path, self.report)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_moved_proposal_uses_original_review(self):
        actual = inherited_review_decisions(self.current, self.value, self.root)
        self.assertEqual(actual, self.source.parent / 'decisions.json')
        self.assertEqual(json.loads(actual.read_text()), {'source': 'original'})

    def test_original_proposal_keeps_local_reference(self):
        self.assertEqual(inherited_review_decisions(self.source, {'agentic_review': self.review}, self.root),
                         self.source.parent / 'decisions.json')

    def test_changed_source_rejected(self):
        self.source.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'source proposal changed'):
            inherited_review_decisions(self.current, self.value, self.root)

    def test_report_for_other_output_rejected(self):
        self.report['outputs']['proposal']['sha256'] = '0' * 64
        self.write(self.report_path, self.report)
        with self.assertRaisesRegex(ValueError, 'does not identify'):
            inherited_review_decisions(self.current, self.value, self.root)

    def test_changed_inherited_review_rejected(self):
        self.value['agentic_review'] = {'decisions_file': 'other.json'}
        self.write(self.current, self.value)
        self.report['outputs']['proposal']['sha256'] = digest_file(self.current)
        self.write(self.report_path, self.report)
        with self.assertRaisesRegex(ValueError, 'differs from its frozen source'):
            inherited_review_decisions(self.current, self.value, self.root)


if __name__ == '__main__':
    unittest.main()
