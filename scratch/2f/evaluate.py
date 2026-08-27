import sys, time, random
sys.path.insert(0, 'scratch/2f')
from dset import DSet
import harness as H, selector as S
from collections import defaultdict

BIG = 10**9

def testset():
    T = {}
    T['causal'] = ('zoo', DSet([(0, BIG)]))
    for w in (128, 256, 512, 1024, 4096):
        T[f'window-{w}'] = ('zoo', DSet([(0, w)]))
    for s in (2, 4, 8):
        T[f'dilated-{s}'] = ('zoo', DSet(aps=[(0, s, 0, BIG)]))
    T['local256+str8'] = ('zoo', DSet([(0, 256)], [(0, 8, 0, BIG)]))
    T['prefix-ish'] = ('zoo', DSet([(0, 64), (0, BIG)]))
    # b5 adversarial
    T['twoband-aligned'] = ('adv', DSet([(0, 128), (1024, 1152)]))
    T['twoband-misaligned'] = ('adv', DSet([(0, 128), (1000, 1128)]))
    T['C2-splitter'] = ('adv', DSet([(0, 24), (500, 524)], [(0, 2, 0, BIG)]))
    # misaligned band variants -- offsets deliberately not multiples of 128
    for off in (37, 91, 205, 611):
        T[f'band-off{off}'] = ('misaligned', DSet([(0, 96), (off, off + 96)]))
    for w in (100, 300, 1000):
        T[f'window-{w}-odd'] = ('misaligned', DSet([(0, w)]))
    for s in (3, 5, 7):
        T[f'dilated-{s}-odd'] = ('misaligned', DSet(aps=[(0, s, 0, BIG)]))
    # random diagonally-invariant D, seeded so all three of us hit identical instances
    for seed in range(6):
        rng = random.Random(1000 + seed)
        offs = sorted(rng.sample(range(0, 2048), 300))
        T[f'random-{seed}'] = ('random', DSet([(o, o + 1) for o in offs]))
    return T


def run(Ns=(1024, 1536, 2048), grid=(128, 64, 32, 16), margin=0.01):
    """Reports overall AND restricted to contested cells, per b5: on a mask where
    every candidate costs the same, any answer 'agrees' and the number is free.
    A cell is contested when oracle best and second-best differ by > margin."""
    T = testset()
    agree = defaultdict(lambda: [0, 0]); regret = defaultdict(list)
    cagree = defaultdict(lambda: [0, 0]); cregret = defaultdict(list)
    declined = defaultdict(int); permask = defaultdict(list)
    for N in Ns:
        for nm, (cat, D) in T.items():
            for BQ in grid:
                for A in grid:
                    o = H.oracle(D, N, BQ, A)
                    if not o:
                        continue
                    pick, c = S.select(D, N, BQ, A)
                    # candidates the oracle could cost but the selector declined
                    declined[cat] += len(set(o) - set(c))
                    vals = sorted(o.values())
                    best = vals[0]
                    second = next((v for v in vals if v > best), None)
                    r = o[pick] / best if best else 1.0
                    ok = o[pick] == best
                    agree[cat][1] += 1; agree[cat][0] += ok
                    regret[cat].append(r); permask[nm].append(r)
                    if second is not None and (second - best) / best > margin:
                        cagree[cat][1] += 1; cagree[cat][0] += ok
                        cregret[cat].append(r)
    return agree, regret, cagree, cregret, declined, permask


if __name__ == '__main__':
    t0 = time.time()
    agree, regret, cagree, cregret, declined, permask = run()
    def block(title, ag, rg):
        print(f'\n{title}')
        print(f'{"category":<13}{"agreement":>13}{"mean regret":>13}{"max regret":>12}')
        tot=[0,0]; allr=[]
        for cat in ('zoo','adv','misaligned','random'):
            a,n = ag[cat]; r = rg[cat]
            if not n: continue
            tot[0]+=a; tot[1]+=n; allr+=r
            print(f'{cat:<13}{a}/{n:<8}{100*a/n:>6.1f}%{sum(r)/len(r):>10.4f}{max(r):>12.4f}')
        if tot[1]:
            print(f'{"ALL":<13}{tot[0]}/{tot[1]:<8}{100*tot[0]/tot[1]:>6.1f}%'
                  f'{sum(allr)/len(allr):>10.4f}{max(allr):>12.4f}')
    block('=== ALL CELLS ===', agree, regret)
    block('=== CONTESTED ONLY (oracle best vs 2nd differ > 1%) ===', cagree, cregret)
    print('\ndeclined candidates (oracle could cost, selector could not):',
          {k: v for k, v in declined.items()})
    worst = sorted(((max(v), k) for k, v in permask.items()), reverse=True)[:6]
    print('worst per-mask max regret:', [(k, round(m,4)) for m,k in worst])
    print(f'\n({time.time()-t0:.0f}s)')
