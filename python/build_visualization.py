#!/usr/bin/env python3
"""Select BoltzGen designs and build a portable ChimeraX bundle (stdlib only)."""
import argparse
import csv
import hashlib
import json
import math
import operator
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

OPS = {'<': operator.lt, '<=': operator.le, '>': operator.gt,
       '>=': operator.ge, '==': operator.eq, '!=': operator.ne}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def truth(value):
    value = str(value).strip().lower()
    if value in ('true', '1'): return True
    if value in ('false', '0'): return False
    return None


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Run directory or its output directory')
    parser.add_argument('destination', type=Path, help='New bundle directory')
    parser.add_argument('--metrics', type=Path, help='Explicit metrics CSV, e.g. refiltered results')
    parser.add_argument('--select', choices=['all', 'passed', 'ids', 'custom'], default='all')
    parser.add_argument('--ids', nargs='+', help='Exact design IDs, space separated')
    parser.add_argument('--criteria', type=Path, help='JSON list of {column, op, value}; all must match')
    parser.add_argument('--sort-by', default='final_rank', help='Numeric metric; missing values sort last')
    parser.add_argument('--descending', action='store_true')
    parser.add_argument('--limit', type=int, default=30, help='Maximum pairs, default 30; 0 = unlimited')
    parser.add_argument('--dry-run', action='store_true', help='Print selection summary without copying')
    args = parser.parse_args(argv)
    if args.limit < 0: parser.error('--limit must be >= 0')
    if (args.select == 'ids') != bool(args.ids): parser.error('--ids is required only with --select ids')
    if (args.select == 'custom') != bool(args.criteria): parser.error('--criteria is required only with --select custom')
    run = args.run.resolve()
    if (run / 'output').is_dir() and not (run / 'intermediate_designs_inverse_folded').is_dir():
        run = run / 'output'
    metrics = (args.metrics or run / 'final_ranked_designs/all_designs_metrics.csv').resolve()
    with metrics.open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows: raise ValueError('Metrics CSV contains no designs')
    if 'id' not in fields: raise ValueError('Metrics CSV must contain an id column')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate design IDs in metrics CSV')
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', x) or x in ('.', '..') for x in ids):
        raise ValueError('Unsafe or empty design ID')
    if args.sort_by not in fields: raise ValueError(f'Unknown sort column: {args.sort_by}')
    if not any(number(r[args.sort_by]) is not None for r in rows):
        raise ValueError('Sort column contains no finite numeric values')
    if args.select == 'passed' and 'pass_filters' not in fields:
        raise ValueError('pass_filters column missing')
    wanted = set(args.ids or [])
    if wanted - set(ids): raise ValueError('Unknown IDs: ' + ', '.join(sorted(wanted - set(ids))))
    criteria = []
    if args.criteria:
        criteria = json.loads(args.criteria.read_text())
        if not isinstance(criteria, list) or not criteria: raise ValueError('Criteria must be a nonempty JSON list')
        for c in criteria:
            if not isinstance(c, dict) or set(c) != {'column', 'op', 'value'}:
                raise ValueError('Each criterion needs exactly column, op, value')
            if c['column'] not in fields: raise ValueError(f"Unknown metric: {c['column']}")
            if c['op'] not in OPS: raise ValueError(f"Unknown operator: {c['op']}")
            if isinstance(c['value'], bool):
                if c['op'] not in ('==', '!='): raise ValueError('Booleans support == and != only')
            elif not isinstance(c['value'], (float, int)) or number(c['value']) is None:
                raise ValueError('Criteria values must be finite numbers or booleans')
    def matches(row):
        for c in criteria:
            value = truth(row[c['column']]) if isinstance(c['value'], bool) else number(row[c['column']])
            if value is None or not OPS[c['op']](value, c['value']): return False
        return True
    eligible = [r for r in rows if (
        args.select == 'all' or
        (args.select == 'passed' and truth(r.get('pass_filters')) is True) or
        (args.select == 'ids' and r['id'] in wanted) or
        (args.select == 'custom' and matches(r)))]
    def key(row):
        value = number(row[args.sort_by])
        return (value is None, (-value if args.descending else value) if value is not None else 0, row['id'])
    eligible.sort(key=key)
    selected = eligible[:args.limit] if args.limit else eligible
    pre = run / 'intermediate_designs_inverse_folded'
    post = pre / 'refold_cif'
    pairs = []
    for row in selected:
        # Prefer recorded filename, preserving exact basename; no rank-prefix guesses.
        name = row.get('file_name') or row['id'] + '.cif'
        if Path(name).name != name or not name.endswith('.cif'):
            raise ValueError(f'Invalid file_name: {name}')
        before, after = pre / name, post / name
        if not before.is_file() or not after.is_file():
            raise ValueError(f"Missing selected structure pair for {row['id']}: {before}, {after}")
        pairs.append(dict(design=row['id'], before=f"structures/{row['id']}/before.cif",
                          after=f"structures/{row['id']}/after.cif",
                          source_before=str(before), source_after=str(after),
                          pass_filters=row.get('pass_filters', 'unknown'),
                          final_rank=row.get('final_rank', ''),
                          sha256_before=digest(before), sha256_after=digest(after)))
    filter_counts = {}
    for col in fields:
        if col.startswith('pass_'):
            counts = Counter(truth(r[col]) for r in rows)
            filter_counts[col] = {'passed': counts[True], 'failed': counts[False], 'unknown': counts[None]}
    summary = dict(total_metrics_rows=len(rows), eligible=len(eligible), selected=len(selected),
                   selection=args.select, limit=args.limit, sort_by=args.sort_by,
                   descending=args.descending, criteria=criteria, requested_ids=sorted(wanted),
                   metrics_source=str(metrics), metrics_sha256=digest(metrics),
                   run_source=str(run), filter_counts=filter_counts,
                   selected_ids=[r['id'] for r in selected],
                   note='Selection is not experimental validation. No fallback to failed designs.')
    print(json.dumps(summary, indent=2))
    if args.dry_run: return 0
    dest = args.destination.resolve()
    if dest.exists(): raise ValueError(f'Destination exists: {dest}; choose a new directory')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.visualization-', dir=dest.parent))
    try:
        for pair in pairs:
            for stage in ('before', 'after'):
                target = temp / pair[stage]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pair['source_' + stage], target)
        (temp / 'pairs.json').write_text(json.dumps(pairs, indent=2) + '\n')
        (temp / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        write_csv(temp / 'selected_metrics.csv', selected, fields)
        manifest_fields = list(pairs[0]) if pairs else ['design', 'before', 'after', 'pass_filters']
        write_csv(temp / 'manifest.csv', pairs, manifest_fields)
        if pairs:
            shutil.copy2(Path(__file__).parent / 'chimerax/compare_refolds.py', temp / 'compare_refolds.py')
        (temp / 'README.txt').write_text(
            f'{len(pairs)} selected pairs from {len(rows)} metrics rows.\n'
            'Read summary.json for exact selection and filter counts.\n'
            + ('Open compare_refolds.py in an empty ChimeraX session.\n'
               'Cyan=before; magenta=after; grey=first selected pre-refold receptor.\n'
               'Toggle one design group at a time. Reference is not the actual receptor for every pair.\n'
               'Check clashes with the matching receptor; pocket highlighting is not implemented.\n'
               if pairs else 'No designs selected; no viewer or structures created. No fallback applied.\n'))
        temp.rename(dest)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    print(f'Bundle: {dest}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        sys.exit(f'Error: {error}')
