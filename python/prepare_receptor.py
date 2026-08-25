"""Prepare a crystal structure as an AutoDock Vina receptor.

Strips waters and ligands, keeps one chain, repairs the modified residues
that break Meeko, and converts to PDBQT.

usage:   python prepare_receptor.py STRUCTURE.cif OUT_PREFIX [chain] [pocket]
example: python prepare_receptor.py data/1P3D.cif murc A 25,27,28,29

Passing the pocket residue list is optional but recommended: dropped
residues are then checked against it, and any that line the binding site
are reported prominently rather than left in the log.

If no chain is given, the one with the most standard residues is used.
Writes OUT_PREFIX.pdb and OUT_PREFIX.pdbqt.

Three repairs are applied, each corresponding to a failure seen in practice:

  MSE (selenomethionine)  renamed to MET, SE renamed to SD and its element
      set to sulfur. Meeko has no covalent radius for selenium and aborts
      with "Element Se doesn't have an implemented covalent radius".

  KCX (carbamylated lysine)  renamed to LYS *and* the carbamyl atoms
      deleted. Renaming alone leaves excess atoms and template matching
      still fails.

  alternate conformations  removed, keeping the primary conformer. Meeko
      otherwise refuses with a list of residues having altlocs.

Residues that remain unmatched (typically incomplete side chains in the
crystal) are dropped by Meeko via --allow_bad_res. Check which ones, and
confirm they are not in the binding site.
"""
import sys
import subprocess
import gemmi

AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO "
         "SER THR TRP TYR VAL".split())

# atoms present in KCX but not in LYS
KCX_EXTRA = {"CX", "OQ1", "OQ2", "OX1", "OX2", "C1", "O1", "O2"}


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    path, prefix = sys.argv[1], sys.argv[2]
    want_chain = sys.argv[3] if len(sys.argv) > 3 else None
    pocket = set()
    if len(sys.argv) > 4:
        pocket = {int(x) for x in sys.argv[4].strip().lstrip("[").rstrip("]")
                  .replace(" ", "").split(",")}

    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_ligands_and_waters()
    st.remove_hydrogens()
    st.remove_alternative_conformations()

    model = st[0]
    if want_chain is None:
        keep = max(model, key=lambda c: sum(1 for r in c if r.name in AA)).name
    else:
        keep = want_chain
    for name in [c.name for c in model]:
        if name != keep:
            model.remove_chain(name)

    n_mse = n_kcx = 0
    for ch in model:
        for r in ch:
            if r.name == "MSE":
                r.name = "MET"
                r.het_flag = "A"
                n_mse += 1
                for a in r:
                    if a.name == "SE":
                        a.name = "SD"
                        a.element = gemmi.Element("S")
            elif r.name == "KCX":
                r.name = "LYS"
                r.het_flag = "A"
                n_kcx += 1
                for aname in [a.name for a in r]:
                    if aname in KCX_EXTRA:
                        del r[[a.name for a in r].index(aname)]

    st.setup_entities()
    pdb = f"{prefix}.pdb"
    st.write_pdb(pdb)

    nres = sum(1 for ch in model for r in ch if r.name in AA)
    print(f"chain kept   : {keep}  ({nres} standard residues)")
    print(f"MSE -> MET   : {n_mse}")
    print(f"KCX -> LYS   : {n_kcx}")
    print(f"wrote        : {pdb}")

    print("\nrunning mk_prepare_receptor.py")
    cmd = ["mk_prepare_receptor.py", "-i", pdb, "-o", prefix,
           "-p", "-a", "--default_altloc", "A"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + res.stderr
    for line in out.splitlines():
        if any(k in line for k in ("Files written", "pdbqt", "Ignored",
                                   "failed", "Error", "error")):
            print("  " + line.strip())

    # which residues did Meeko drop, and do any of them line the pocket?
    dropped = set()
    for line in out.splitlines():
        if "Template matching failed" in line:
            import re
            dropped = {int(m) for m in re.findall(r"[A-Za-z]:(\d+)", line)}
    if dropped:
        print(f"  dropped {len(dropped)} unmatched residues "
              f"(usually incomplete side chains)")
        if pocket:
            bad = sorted(dropped & pocket)
            if bad:
                print(f"\n  WARNING: dropped residues in the binding site: {bad}")
                print("  the receptor is incomplete where it matters most.")
                print("  consider a different structure, or model the missing")
                print("  side chains before docking.")
            else:
                print("  none of them are in the given pocket")

    import os
    if os.path.exists(f"{prefix}.pdbqt"):
        print(f"\nreceptor ready: {prefix}.pdbqt")
    else:
        print("\nreceptor preparation FAILED - full output follows\n")
        print(out)
        sys.exit(1)


if __name__ == "__main__":
    main()
