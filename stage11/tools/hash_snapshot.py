"""Hash snapshot of protected artifacts (Stage 1-9, Stage 10, frozen Stage 11 v2 checkpoint).

    python -P stage11/tools/hash_snapshot.py --write stage11/audit/hashes_before.txt
    python -P stage11/tools/hash_snapshot.py --compare stage11/audit/hashes_before.txt
"""
import argparse
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTECTED = ['models/baseline', 'models/contrastive', 'models/stage11/best.pt', 'models/stage11/last.pt',
             'models/stage11/calibration.json', 'models/stage11/v1', 'stage10', 'config.yaml', 'results',
             'data/splits', 'demo/app.py', 'src', 'scripts', 'tests']
SKIP = {'__pycache__', '.pytest_cache'}


def snapshot():
    out = {}
    for r in PROTECTED:
        p = ROOT / r
        files = [p] if p.is_file() else sorted(x for x in p.rglob('*') if x.is_file() and not SKIP & set(x.parts))
        for f in files:
            out[f.relative_to(ROOT).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--write')
    g.add_argument('--compare')
    a = ap.parse_args()
    now = snapshot()
    if a.write:
        pathlib.Path(a.write).write_text(''.join(f'{h}  {k}\n' for k, h in sorted(now.items())), encoding='utf-8')
        print(f'{len(now)} files hashed -> {a.write}')
        return 0
    before = dict(reversed(line.split('  ', 1)) for line in
                  pathlib.Path(a.compare).read_text(encoding='utf-8').splitlines() if line.strip())
    changed = [k for k in before if k in now and now[k] != before[k]]
    missing = [k for k in before if k not in now]
    added = [k for k in now if k not in before]
    print(f'{len(before)} protected files: changed {len(changed)}, missing {len(missing)}, added {len(added)}')
    for k in changed + missing:
        print('  CHANGED/MISSING', k)
    for k in added:
        print('  added', k)
    return 1 if changed or missing else 0


if __name__ == '__main__':
    sys.exit(main())
