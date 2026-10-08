"""Write a BoltzGen design specification targeting pocket residues.

Usage:
    python make_spec.py STRUCTURE.cif CHAIN RESIDUES LENGTH OUT.yaml
    python make_spec.py data/1P3D.cif A 25,27,28,29 8..16 murc.yaml --retain-chains F

CHAIN and --retain-chains use canonical mmCIF IDs (label_asym_id).
RESIDUES uses canonical indices (label_seq_id) from map_pocket.py.
LENGTH is a positive integer or range, e.g. 12 or 8..16.

Additional components are imported from the same structure file.
They are not added to the binding-site guidance. The designed peptide
defaults to ID B; use --peptide-chain to avoid included-ID conflicts.

This script checks argument formatting and ID conflicts, but does not
validate IDs or residue indices against the CIF. Run boltzgen check
and inspect its output before generation.
"""
import argparse
import json
import os
import re

TEMPLATE = """entities:
  - protein:
      id: {peptide_chain}
      sequence: {length}
  - file:
      path: {path}
      include:
{included_chains}
      # Binding guidance belongs to this file entity.
      binding_types:
        - chain:
            id: {chain}
            binding: {residues}
"""
def main():
    """Read selections and write the YAML without changing the source CIF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Source mmCIF file")
    parser.add_argument("chain", help="Canonical target protein ID")
    parser.add_argument("residues", help="Comma-separated canonical pocket indices")
    parser.add_argument("length", help="Peptide length, e.g. 12 or 8..16")
    parser.add_argument("out", help="Output YAML file")
    parser.add_argument("--retain-chains", default="",
                        help="Additional canonical IDs, e.g. F or F,C,D")
    parser.add_argument("--peptide-chain", default="B",
                        help="New peptide ID (default: B)")
    args = parser.parse_args()

    # Accept a bracketed residue list or the plain output from map_pocket.py.
    residues = args.residues.strip().lstrip("[").rstrip("]").replace(" ", "")
    residue_list = residues.split(",")
    if any(not value.isdigit() or int(value) < 1 for value in residue_list):
        parser.error("Pocket indices must be comma-separated positive integers.")

    if not re.fullmatch(r"[1-9][0-9]*(\.\.[1-9][0-9]*)?", args.length):
        parser.error("Length must be a positive integer or range, e.g. 8..16.")
    bounds = [int(value) for value in args.length.split("..")]
    if bounds[0] > bounds[-1]:
        parser.error("The minimum peptide length exceeds the maximum.")

    retained = (
        [value.strip() for value in args.retain_chains.split(",")]
        if args.retain_chains else []
    )
    included = [args.chain] + retained
    if any(not value or value in {"?", "."} or
           any(character.isspace() for character in value) or "," in value
           for value in included + [args.peptide_chain]):
        parser.error("Each chain ID must be a nonempty identifier without spaces or commas.")
    if len(set(included)) != len(included):
        parser.error("Include each canonical ID only once, including the target.")
    if args.peptide_chain in included:
        parser.error("The peptide ID conflicts with an included ID; use --peptide-chain.")

    # An absolute source path works even when the YAML is saved elsewhere.
    abspath = os.path.abspath(args.path)
    if not os.path.isfile(abspath):
        parser.error(f"Structure file not found: {abspath}")
    if os.path.realpath(args.out) == os.path.realpath(abspath):
        parser.error("The output must not overwrite the source structure.")

    # JSON-quoted strings are valid YAML and preserve numeric-looking IDs.
    included_chains = "\n".join(
        f"        - chain:\n            id: {json.dumps(chain_id)}"
        for chain_id in included
    )
    with open(args.out, "w") as fh:
        fh.write(TEMPLATE.format(
            path=json.dumps(abspath),
            chain=json.dumps(args.chain),
            peptide_chain=json.dumps(args.peptide_chain),
            residues=json.dumps(residues),
            length=json.dumps(args.length),
            included_chains=included_chains,
        ))

    print(f"wrote {args.out}")
    print(f"  structure : {abspath}")
    print(f"  target    : {args.chain}")
    print(f"  retained  : {', '.join(retained) or 'none'}")
    print(f"  peptide   : {args.peptide_chain}")
    print(f"  binding   : {len(set(residue_list))} unique residues")
    print(f"  length    : {args.length}")
    print(f"\nnext: boltzgen check {args.out}")


if __name__ == "__main__":
    main()

