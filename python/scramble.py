"""Generate composition-matched scrambles of a peptide sequence.

usage:   python scramble.py SEQUENCE [n] [seed]
example: python scramble.py TMGFTAPRFPHY 5 11

Scrambles have identical amino acid composition and differ only in order,
so they control for everything except sequence. This is the most
informative control in this workflow: an apparent result that a scramble
reproduces carries no sequence information.

Two findings here came from such controls. Designed peptides appeared to
prefer MurC's ATP pocket over its substrate pocket by 1.38 kcal/mol, but
scrambles preferred it by 1.59, so the effect was pocket enclosure rather
than sequence. And a validated 250 uM MurF inhibitor ranked last of six
against five scrambles of itself.

Sequences are printed on stdout, so the output can be passed straight to
06_prepare_ligands.sh.
"""
import sys
import random


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    seq = sys.argv[1].upper()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    rng = random.Random(seed)
    seen = {seq}
    out = []
    for _ in range(2000):
        if len(out) >= n:
            break
        s = "".join(rng.sample(seq, len(seq)))
        if s not in seen:
            seen.add(s)
            out.append(s)

    if len(out) < n:
        print(f"# only {len(out)} distinct scrambles possible", file=sys.stderr)
    print(f"# original {seq}, seed {seed}", file=sys.stderr)
    print(" ".join(out))


if __name__ == "__main__":
    main()
