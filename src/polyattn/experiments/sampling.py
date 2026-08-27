"""Quantify the row-block sampling error at the granularities the sweep samples."""
from .. import cost as model, masks

DEFAULT = [masks.SlidingWindow(128), masks.DocPacked(512), masks.Dilated(8),
           masks.SinksWindow(4, 256), masks.LocalStrided(256, 8)]


def run(N=16384, zoo=None, grains=((32, 32), (16, 16))):
    zoo = zoo or DEFAULT
    out = []
    print(f"{'mask':<16}{'BQ':>4}{'A':>4}{'sampled':>14}{'exact':>14}{'rel.err':>10}")
    for m in zoo:
        for BQ, A in grains:
            s, _ = model.cost(m, N, BQ, A)
            e, _ = model.cost(m, N, BQ, A, exact_only=True)
            err = abs(s - e) / e * 100
            out.append((m.name, BQ, A, err))
            print(f"{m.name:<16}{BQ:>4}{A:>4}{s:>14.4g}{e:>14.4g}{err:>9.4f}%")
    print(f"\nworst relative error: {max(o[3] for o in out):.4f}%")
    return out


if __name__ == "__main__":
    run()
