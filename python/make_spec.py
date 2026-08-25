"""Write a BoltzGen design spec targeting a set of pocket residues.

usage:   python make_spec.py STRUCTURE.cif CHAIN RESIDUES LENGTH OUT.yaml
example: python make_spec.py data/1P3D.cif A 25,27,28,29 8..16 murc.yaml

RESIDUES is the comma-separated list printed by map_pocket.py.
LENGTH is a BoltzGen range, e.g. 8..16 or 12..18.

Note that BoltzGen indexes by label_seq_id, which does not always match the
author numbering that map_pocket.py reports. Always run `boltzgen check` on
the generated spec and confirm the binding site is where you intended.
"""
import sys
import os

TEMPLATE = """entities:
  - protein:
      id: B
      sequence: {length}
  - file:
      path: {path}
      include:
        - chain:
            id: {chain}
binding_types:
  - chain:
      id: {chain}
      binding: {residues}
"""


def main():
    if len(sys.argv) < 6:
        sys.exit(__doc__)

    path, chain, residues, length, out = sys.argv[1:6]

    # accept the bracketed list that map_pocket.py prints, or a plain one
    residues = residues.strip().lstrip("[").rstrip("]").replace(" ", "")
    nres = len(residues.split(","))

    # BoltzGen resolves the structure path relative to the working directory,
    # so store it as an absolute path to avoid surprises
    abspath = os.path.abspath(path)

    with open(out, "w") as fh:
        fh.write(TEMPLATE.format(path=abspath, chain=chain,
                                 residues=residues, length=length))

    print(f"wrote {out}")
    print(f"  structure : {abspath}")
    print(f"  chain     : {chain}")
    print(f"  binding   : {nres} residues")
    print(f"  length    : {length}")
    print(f"\nnext: boltzgen check {out}")


if __name__ == "__main__":
    main()
