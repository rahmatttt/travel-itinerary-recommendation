"""Run every optimizer class on synthetic data and dump a normalized trace to JSON.

Compare the traces taken before and after a change with compare_traces.py.
Usage: python tests/run_trace.py <out.json>
"""
import ast
import datetime
import glob
import hashlib
import importlib
import inspect
import json
import os
import random
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from optimization.node import Node

SKIP = {'koneksi', 'node', '__init__', 'base'}
SEEDS = [7, 42]
T = datetime.time


def norm(x, seen=None):
    """Convert a value to a JSON-friendly form that keeps exact float values."""
    seen = seen or set()
    if isinstance(x, (int, float, str, bool, type(None))):
        return repr(x)
    if isinstance(x, np.generic):
        return repr(x.item())
    if isinstance(x, np.ndarray):
        return ['nd', [norm(i, seen) for i in x.tolist()]]
    if isinstance(x, (datetime.time, datetime.datetime)):
        return str(x)
    if id(x) in seen:
        return '<cycle>'
    seen = seen | {id(x)}
    if isinstance(x, dict):
        return {repr(k): norm(v, seen) for k, v in sorted(x.items(), key=lambda kv: repr(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [norm(i, seen) for i in x]
    if isinstance(x, set):
        return sorted(repr(i) for i in x)
    if callable(x):
        return '<callable>'
    if hasattr(x, '__dict__'):
        return {type(x).__name__: norm({k: v for k, v in vars(x).items() if k != 'db'}, seen)}
    return repr(x)


def rng_state():
    s = np.random.get_state()
    return hashlib.md5(repr(random.getstate()).encode() + s[1].tobytes() + str(s[2]).encode()).hexdigest()


def tour_data():
    r = random.Random(1)
    nodes = [Node(i, f'p{i}', -7.7 + i / 100, 110.3 + i / 100, 1800 + 300 * (i % 4), 3.5 + r.random(), 5000 * (i % 5),
                  'wisata', T(7, 0, 0) if i % 3 else T(9, 0, 0), T(18, 0, 0) if i % 4 else T(21, 0, 0)) for i in range(1, 9)]
    hotel = Node(100, 'hotel', -7.8, 110.4, 0, 4.0, 0, 'hotel', T(0, 0, 0), T(23, 59, 0))
    ids = [x._id for x in nodes] + [hotel._id]
    return nodes, hotel, ids, r


def timematrix(ids, r):
    return {a: {b: {'waktu': 0 if a == b else r.randint(600, 3600)} for b in ids} for a in ids}


def td_matrix(ids, r, day_types):
    return {d: {a: {b: {h: (0 if a == b else r.randint(600, 3600)) for h in range(24)} for b in ids} for a in ids} for d in day_types}


def solomon():
    r = random.Random(2)
    depot = {'id': 0, 'x': 50.0, 'y': 50.0, 'd': 0, 'S': 0, 'O': 0, 'C': 400, 'b': 0, 't': 0, 'a': 0, 'f': 0, 'e': 0}
    nodes = [{'id': i, 'x': r.uniform(0, 100), 'y': r.uniform(0, 100), 'd': 10, 'S': r.randint(1, 30), 'O': r.randint(0, 100),
              'C': r.randint(150, 390), 'b': r.randint(1, 30), 't': r.randint(1, 30), 'a': r.randint(1, 30), 'f': r.randint(1, 30), 'e': r.randint(1, 30)}
             for i in range(1, 11)]
    return nodes, depot


def small_kwargs(cls):
    """Shrink iteration and population parameters so each run stays fast."""
    out = {}
    for n, p in inspect.signature(cls.__init__).parameters.items():
        if n == 'self' or p.default is inspect.Parameter.empty or not isinstance(p.default, (int, float)) or isinstance(p.default, bool):
            continue
        low = n.lower()
        if 'idem' in low:
            out[n] = 2
        elif 'iter' in low or low in ('generations', 'max_gen'):
            out[n] = 3
        elif 'pop' in low or low in ('n_ants', 'num_ants', 'ants', 'n_particles', 'n_wolves', 'n_whales', 'n_bee', 'n_fireflies', 'ant', 'colony'):
            out[n] = 6
    return out


def trace_class(d, cls, seed, kw):
    random.seed(0)
    np.random.seed(0)
    tr = {}
    try:
        init_kw = dict(kw)
        if 'random_state' in inspect.signature(cls.__init__).parameters:
            init_kw['random_state'] = seed
        obj = cls(**init_kw)
        tr['init_vars'] = norm({k: v for k, v in vars(obj).items() if k != 'db'})
        tr['has_db'] = hasattr(obj, 'db')
        tr['init_rng'] = rng_state()
        if d == 'optimization' or 'time_dependent' in d:
            nodes, hotel, ids, r = tour_data()
            if 'time_dependent' in d:
                tm = td_matrix(ids, r, ['weekday', 'weekend', 0, 1, 2, 3, 4, 5, 6, 'senin', 'sabtu', 'minggu'])
            else:
                tm = timematrix(ids, r)
            obj.set_model(nodes, hotel, tm, travel_days=2, degree_waktu=1, degree_tarif=seed % 2, degree_rating=1)
        else:
            nodes, depot = solomon()
            obj.set_model(nodes, depot, num_vehicle=2)
        tr['model_vars'] = norm({k: v for k, v in vars(obj).items() if k != 'db'})
        tr['model_rng'] = rng_state()
        tr['result'] = norm(obj.construct_solution())
        tr['final_rng'] = rng_state()
    except Exception as e:
        tr['error'] = f'{type(e).__name__}: {e}'
        tr['error_tb'] = traceback.format_exc().splitlines()[-3:]
    return tr


def main():
    out_path = os.path.abspath(sys.argv[1])
    os.chdir(ROOT)
    results = {}
    for d in sorted(p for p in glob.glob('optimization*') if os.path.isdir(p)):
        for f in sorted(glob.glob(os.path.join(d, '*.py'))):
            m = os.path.basename(f)[:-3]
            if m in SKIP:
                continue
            mod = importlib.import_module(f'{d}.{m}')
            names = [c.name for c in ast.parse(open(f, encoding='utf-8').read()).body if isinstance(c, ast.ClassDef)]
            for cn in names:
                cls = getattr(mod, cn)
                kw = small_kwargs(cls)
                for seed in SEEDS:
                    key = f'{d}.{m}.{cn}@{seed}'
                    t0 = time.time()
                    results[key] = trace_class(d, cls, seed, kw)
                    print(key, round(time.time() - t0, 2), results[key].get('error', 'ok'), flush=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, sort_keys=True)
    errors = [k for k, v in results.items() if 'error' in v]
    print(f'{len(results)} runs, {len(errors)} errors')
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
