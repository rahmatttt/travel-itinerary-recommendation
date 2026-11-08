"""Compare two traces written by run_trace.py. Exits with 1 if they differ.

Usage: python tests/compare_traces.py <before.json> <after.json>
"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')


def main():
    with open(sys.argv[1], encoding='utf-8') as fh:
        before = json.load(fh)
    with open(sys.argv[2], encoding='utf-8') as fh:
        after = json.load(fh)
    missing = sorted(set(before) ^ set(after))
    changed = [k for k in sorted(set(before) & set(after)) if before[k] != after[k]]
    print(f'{len(before)} runs before, {len(after)} after, {len(changed)} changed, {len(missing)} only in one trace')
    for k in missing:
        print('only in one trace:', k)
    for k in changed:
        fields = sorted(f for f in set(before[k]) | set(after[k]) if before[k].get(f) != after[k].get(f))
        print(k, ','.join(fields))
    sys.exit(1 if changed or missing else 0)


if __name__ == '__main__':
    main()
