import concurrent.futures
import json
import statistics
import time
import urllib.request
from pathlib import Path

OUT = Path('/public/home/u43077/lzh/benchmarks/next-20260801/http-matrix.json')
ENGINES = {'vllm': 18000, 'sglang': 18001}
CASES = [(512, 32), (512, 128), (4096, 32), (4096, 128)]
CONC = [1, 8, 16]

def one(port, prompt, out_len):
    body = json.dumps({'model': 'Qwen3-8B', 'prompt': prompt, 'max_tokens': out_len, 'temperature': 0, 'top_p': 1, 'stream': False, 'ignore_eos': True}).encode()
    req = urllib.request.Request('http://127.0.0.1:%d/v1/completions' % port, data=body, headers={'Content-Type': 'application/json'})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        p = json.loads(r.read())
    return time.perf_counter() - t, p.get('usage', {})

def run(engine, port, inp, out, c):
    prompt = ' benchmark' * inp
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        t = time.perf_counter()
        rows = list(ex.map(lambda _: one(port, prompt, out), range(max(c, 8))))
    wall = time.perf_counter() - t
    lat = [x[0] for x in rows]
    ti = sum(x[1].get('prompt_tokens', 0) for x in rows)
    to = sum(x[1].get('completion_tokens', 0) for x in rows)
    return {'engine': engine, 'input_target': inp, 'output_target': out, 'concurrency': c, 'requests': len(rows), 'wall_s': wall, 'input_tokens': ti, 'output_tokens': to, 'input_tps': ti / wall, 'output_tps': to / wall, 'p50_s': statistics.median(lat), 'p95_s': sorted(lat)[max(0, int(.95 * len(lat)) - 1)], 'errors': 0}

results = []
for engine, port in ENGINES.items():
    for inp, out in CASES:
        for c in CONC:
            try:
                row = run(engine, port, inp, out, c)
            except Exception as e:
                row = {'engine': engine, 'input_target': inp, 'output_target': out, 'concurrency': c, 'errors': 1, 'error': repr(e)}
            print(json.dumps(row), flush=True)
            results.append(row)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({'results': results}, indent=2) + '\n')
