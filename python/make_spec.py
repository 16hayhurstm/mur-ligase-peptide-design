"""Write a BoltzGen design specification (.yaml) targeting pocket residues.

usage:   python make_spec.py STRUCTURE.cif CHAIN RESIDUES LENGTH OUT.yaml
example: python make_spec.py data/1P3D.cif A 25,27,28,29 8..16 murc.yaml

CHAIN is the canonical mmCIF chain identifier (label_asym_id).
RESIDUES is a comma-separated list of canonical residue indices
(label_seq_id). Use the chain and residue list labelled "BoltzGen residues"
in the output from map_pocket.py.
LENGTH is the peptide length or length range, e.g. 12 or 8..16.

Canonical identifiers can differ from author chain names and residue
numbers. This script writes the supplied identifiers without converting
or validating them, so do NOT use the author-numbered list.

Always run `boltzgen check` on the generated specification and inspect
the resulting structure to confirm that the intended pocket is marked.
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
      # Binding guidance belongs to this file entity.
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
