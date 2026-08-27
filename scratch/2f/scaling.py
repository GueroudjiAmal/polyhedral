import sys, time
sys.path.insert(0, 'scratch/2f')
from dset import DSet
import harness as H, selector as S

BIG = 10**9
D = DSet([(0, 256)], [(0, 8, 0, BIG)])          # local256+str8
print(f'{"N":>7}{"selector(s)":>13}{"oracle(s)":>12}{"speedup":>10}')
prev = None
for N in (512, 1024, 2048, 4096, 8192):
    t = time.time(); S.select(D, N, 128, 32); ts = time.time() - t
    if N <= 4096:
        t = time.time(); H.oracle(D, N, 128, 32); to = time.time() - t
    else:
        to = float('nan')
    r = f'{ts/prev:.2f}x' if prev else '--'
    print(f'{N:>7}{ts:>13.4f}{to:>12.4f}{to/ts:>9.1f}x   selector growth vs prev N: {r}')
    prev = ts
