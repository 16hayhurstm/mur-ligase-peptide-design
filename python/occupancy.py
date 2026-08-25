"""Measure how much of each designed peptide sits in the target pocket.

For every design-stage complex, computes the fraction of its protein
contacts that fall within the pocket residue set it was designed against.

usage:   python occupancy.py DESIGN_DIR RESIDUES [cutoff]
example: python occupancy.py out_murc/intermediate_designs 25,27,28,29 5.0

Use the design-stage structures (intermediate_designs/), not the refolded
complexes. BoltzGen's own ranking is computed on an independent refold that
does not preserve the designed placement: in this project 49/100 designs
exceeded 30% occupancy at the design stage and 0/20 did after refolding.
Ranking on refold-derived scores therefore mis-measures the site the design
was aimed at.
"""
import sys
import glob
import os
import gemmi
import numpy as np


def analyse(path, pocket, cutoff):
    st = gemmi.read_structure(path)
    st.setup_entities()
    chains = sorted(st[0], key=lambda c: len(c))
    if len(chains) < 2:
        return None
    pep, prot = chains[0], chains[-1]

    seq = gemmi.one_letter_code([r.name for r in pep]).upper().replace("X", "")
    pep_xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in pep for a in r])

    hits = set()
    for r in prot:
        xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r])
        if np.linalg.norm(xyz[:, None] - pep_xyz[None, :], axis=-1).min() < cutoff:
            hits.add(r.seqid.num)
    if not hits:
        return None

    on = hits & pocket
    return seq, len(seq), len(hits), len(on), len(on) / len(hits)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    design_dir = sys.argv[1]
    pocket = {int(x) for x in
              sys.argv[2].strip().lstrip("[").rstrip("]").replace(" ", "").split(",")}
    cutoff = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    files = sorted(glob.glob(os.path.join(design_dir, "*.cif")))
    if not files:
        sys.exit(f"error: no .cif files in {design_dir}")

    rows = [r for r in (analyse(f, pocket, cutoff) for f in files) if r]
    if not rows:
        sys.exit("error: no analysable complexes - do these files contain "
                 "both chains?")

    rows.sort(key=lambda r: -r[4])
    frac = np.array([r[4] for r in rows])

    print(f"analysed {len(rows)} of {len(files)} structures  "
          f"(pocket = {len(pocket)} residues, cutoff {cutoff} A)\n")
    print(f"{'sequence':<20}{'len':>5}{'contacts':>10}{'in pocket':>11}{'on target':>11}")
    for seq, n, tot, on, f in rows[:20]:
        print(f"{seq:<20}{n:>5}{tot:>10}{on:>11}{f:>10.0%}")
    if len(rows) > 20:
        print(f"... {len(rows) - 20} more")

    print(f"\nmean on-target : {frac.mean():.0%}")
    print(f"  >50%         : {(frac > 0.5).sum()} / {len(frac)}")
    print(f"  >30%         : {(frac > 0.3).sum()} / {len(frac)}")
    print(f"  <10%         : {(frac < 0.1).sum()} / {len(frac)}")


if __name__ == "__main__":
    main()
