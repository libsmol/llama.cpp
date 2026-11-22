#!/usr/bin/env python3
# kldpair.py a:b [c:d ...]: per-chunk paired KLD differences between logs/kld_<a>.txt and logs/kld_<b>.txt
import re, sys, math
def per_chunk(name):
    rows = []
    for l in open(f'/home/hxcliff/src/inkling-work/logs/kld_{name}.txt', errors='replace'):
        m = re.match(r'\s*(\d+)\s+[\d.]+\s+.*?([-\d.]+)\s+±\s+[\d.]+\s+([-\d.]+)\s+±\s+[\d.]+\s+([\d.]+)\s+±', l)
        if m: rows.append((int(m.group(1)), float(m.group(3)), float(m.group(4))))
    kld = [c*n - (rows[k-1][1]*rows[k-1][0] if k else 0.0) for k, (n, c, _) in enumerate(rows)]
    top = [t*n - (rows[k-1][2]*rows[k-1][0] if k else 0.0) for k, (n, _, t) in enumerate(rows)]
    return kld, top
for pair in sys.argv[1:]:
    a, b = pair.split(':')
    (ka, ta), (kb, tb) = per_chunk(a), per_chunk(b)
    n = min(len(ka), len(kb))
    def stat(x, y):
        d = [p - q for p, q in zip(x[:n], y[:n])]; m = sum(d)/n
        se = math.sqrt(sum((v - m)**2 for v in d)/(n - 1))/math.sqrt(n) if n > 1 else float('nan')
        return m, se
    mk, sk = stat(ka, kb); mt, st = stat(ta, tb)
    print(f"{a:24s} - {b:24s}: KLD {mk*1e3:+.3f}e-3 ± {sk*1e3:.3f}e-3 (t={mk/sk if sk else 0:+.2f}) | top1 {mt:+.3f}% ± {st:.3f} (t={mt/st if st else 0:+.2f}) n={n}")
