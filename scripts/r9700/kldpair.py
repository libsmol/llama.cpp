#!/usr/bin/env python3
# kldpair.py a:b [c:d ...]: per-chunk paired KLD and top-1 differences between WORK/logs/kld_<a>.txt and WORK/logs/kld_<b>.txt
import math, os, re, sys

WORK = os.environ.get('WORK', os.path.expanduser('~/src/r9700-work'))

def per_chunk(name):
    rows = []
    for l in open(os.path.join(WORK, 'logs', f'kld_{name}.txt'), errors='replace'):
        m = re.match(r'\s*(\d+)\s+[\d.]+\s+.*?([-\d.]+)\s+±\s+[\d.]+\s+([-\d.]+)\s+±\s+[\d.]+\s+([\d.]+)\s+±', l)
        if m:
            rows.append((int(m.group(1)), float(m.group(3)), float(m.group(4))))
    # the log has running means: back out the value of each chunk
    kld = [c*n - (rows[k-1][1]*rows[k-1][0] if k else 0.0) for k, (n, c, _) in enumerate(rows)]
    top = [t*n - (rows[k-1][2]*rows[k-1][0] if k else 0.0) for k, (n, _, t) in enumerate(rows)]
    return kld, top

def stat(x, y, n):
    d = [p - q for p, q in zip(x[:n], y[:n])]
    m = sum(d)/n
    se = math.sqrt(sum((v - m)**2 for v in d)/(n - 1))/math.sqrt(n) if n > 1 else float('nan')
    return m, se

for pair in sys.argv[1:]:
    a, b = pair.split(':')
    (ka, ta), (kb, tb) = per_chunk(a), per_chunk(b)
    n = min(len(ka), len(kb))
    mk, sk = stat(ka, kb, n)
    mt, st = stat(ta, tb, n)
    print(f"{a:24s} - {b:24s}: KLD {mk*1e3:+.3f}e-3 ± {sk*1e3:.3f}e-3 (t={mk/sk if sk else 0:+.2f}) | "
          f"top1 {mt:+.3f}% ± {st:.3f} (t={mt/st if st else 0:+.2f}) n={n}")
