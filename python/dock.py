"""Dock ligands with AutoDock Vina across several random seeds.

usage:   python dock.py RECEPTOR.pdbqt LIGAND_DIR CX,CY,CZ SIZE [seeds] [cpu]
example: python dock.py murc.pdbqt pep 25.69,-5.73,48.55 24 3 8

Multi-seed by default, and deliberately so. Vina's search is stochastic,
and on this system re-running one peptide against one receptor with
nothing changed but the seed gave a spread of 0.91 kcal/mol, against a
range of 1.29 across ten different peptides. A single-seed number is
therefore not a measurement; the seed spread is reported alongside every
mean so that differences smaller than it are not over-read.

`cpu` limits threads per docking. Vina otherwise spawns one thread per
exhaustiveness pass, which took 32 of 48 cores on a shared machine.
"""
import sys
import os
import glob
import numpy as np
import pandas as pd
from vina import Vina

SEEDS = [1, 42, 137, 2024, 31337]


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)

    receptor = sys.argv[1]
    lig_dir = sys.argv[2]
    centre = [float(x) for x in sys.argv[3].split(",")]
    size = float(sys.argv[4])
    nseed = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    cpu = int(sys.argv[6]) if len(sys.argv) > 6 else 8

    seeds = SEEDS[:nseed]
    ligands = sorted(glob.glob(os.path.join(lig_dir, "*.pdbqt")))
    if not ligands:
        sys.exit(f"error: no .pdbqt files in {lig_dir}")

    print(f"receptor : {receptor}")
    print(f"box      : centre {centre}, size {size} A")
    print(f"ligands  : {len(ligands)}")
    print(f"seeds    : {seeds}\n")

    rows = []
    for lig in ligands:
        name = os.path.basename(lig).replace(".pdbqt", "")
        vals = []
        for sd in seeds:
            v = Vina(sf_name="vina", seed=sd, cpu=cpu)
            v.set_receptor(receptor)
            v.set_ligand_from_file(lig)
            v.compute_vina_maps(center=centre, box_size=[size] * 3)
            v.dock(exhaustiveness=32, n_poses=9)
            e = v.energies(n_poses=9)
            vals.append(e[0][0])
            if sd == seeds[0]:
                out = os.path.join(lig_dir, f"{name}_docked.pdbqt")
                v.write_poses(out, n_poses=9, overwrite=True)
                pose_spread = e[0][0] - e[-1][0]
        vals = np.array(vals)
        rows.append((name, vals.mean(), vals.std(), vals.ptp(),
                     pose_spread, list(np.round(vals, 3))))
        print(f"{name:<24}{vals.mean():8.2f}  sd {vals.std():5.2f}  "
              f"seed range {vals.ptp():5.2f}", flush=True)

    df = pd.DataFrame(rows, columns=["ligand", "mean", "sd", "seed_range",
                                     "pose_spread", "seeds"])
    df = df.sort_values("mean")
    out_csv = os.path.join(lig_dir, "dock_results.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n{df[['ligand','mean','sd','seed_range','pose_spread']].to_string(index=False)}")
    print(f"\nwrote {out_csv}")

    worst = df.seed_range.max()
    span = df["mean"].max() - df["mean"].min()
    print(f"\nspread across ligands : {span:.2f} kcal/mol")
    print(f"largest seed range    : {worst:.2f} kcal/mol")
    if worst > span * 0.5:
        print("\nWARNING: seed variation is comparable to the difference")
        print("between ligands. Treat this as a coarse tier, not a ranking.")


if __name__ == "__main__":
    main()
