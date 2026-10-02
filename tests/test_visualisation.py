"""Selection and copy tests; synthetic file contents are not molecular tests."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / 'python/build_visualisation.py'

class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'run/output'
        self.pre = self.run / 'intermediate_designs_inverse_folded'
        (self.pre / 'refold_cif').mkdir(parents=True)
        self.metrics = self.run / 'final_ranked_designs/all_designs_metrics.csv'
        self.metrics.parent.mkdir()
        self.rows = [dict(id='design_a', file_name='design_a.cif', final_rank=2, pass_filters='False', score='nan'),
                     dict(id='design_b', file_name='design_b.cif', final_rank=1, pass_filters='True', score=0.8)]
        self.write_rows()
        for row in self.rows:
            for folder in (self.pre, self.pre / 'refold_cif'):
                (folder / row['file_name']).write_text('synthetic test fixture\n')
    def write_rows(self):
        with self.metrics.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.rows[0]); w.writeheader(); w.writerows(self.rows)
    def call(self, *args, success=True):
        dest = self.root / 'bundle'
        r = subprocess.run([sys.executable, str(BUILDER), str(self.root/'run'), str(dest), *args], capture_output=True, text=True)
        self.assertEqual(r.returncode == 0, success, r.stderr)
        return dest
    def test_default_order_and_cap(self):
        d = self.call('--limit', '1')
        self.assertEqual(json.loads((d/'pairs.json').read_text())[0]['design'], 'design_b')
        self.assertFalse((d/'structures/design_a').exists())
    def test_passed(self):
        d = self.call('--select', 'passed'); self.assertEqual(len(json.loads((d/'pairs.json').read_text())), 1)
    def test_zero_pass(self):
        self.rows[1]['pass_filters'] = 'False'; self.write_rows()
        d = self.call('--select', 'passed')
        self.assertEqual(json.loads((d/'pairs.json').read_text()), [])
        self.assertFalse((d/'compare_refolds.py').exists())
    def test_explicit_failure_id(self):
        d = self.call('--select', 'ids', '--ids', 'design_a')
        self.assertEqual(json.loads((d/'pairs.json').read_text())[0]['pass_filters'], 'False')
    def test_custom_missing_fails(self):
        c = self.root/'criteria.json'; c.write_text(json.dumps([dict(column='score', op='>=', value=0.5)]))
        d = self.call('--select', 'custom', '--criteria', str(c))
        self.assertEqual(len(json.loads((d/'pairs.json').read_text())), 1)
    def test_unknown_metric(self):
        c = self.root/'criteria.json'; c.write_text(json.dumps([dict(column='typo', op='>=', value=0.5)]))
        self.call('--select', 'custom', '--criteria', str(c), success=False)
    def test_missing_selected_pair(self):
        (self.pre/'refold_cif/design_b.cif').unlink(); self.call(success=False)
    def test_unselected_missing_pair(self):
        (self.pre/'refold_cif/design_a.cif').unlink(); self.call('--select', 'passed')
    def test_unknown_id(self):
        self.call('--select', 'ids', '--ids', 'missing', success=False)
    def test_duplicate_ids(self):
        self.rows[1]['id'] = 'design_a'; self.write_rows(); self.call(success=False)
    def test_dry_run(self):
        d = self.call('--dry-run'); self.assertFalse(d.exists())
    def test_existing_destination(self):
        d = self.call(); marker = d/'keep'; marker.write_text('keep')
        self.call(success=False); self.assertEqual(marker.read_text(), 'keep')

if __name__ == '__main__': unittest.main()
