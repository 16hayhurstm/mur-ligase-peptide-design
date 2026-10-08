"""List observed chains and components in a PDB mmCIF file.

Usage:
    python python/inspect_structure.py data/1P3D.cif

Canonical IDs distinguish structural instances. Shared entity IDs identify
copies of the same molecular entity, not necessarily identical conformations.
Atom records from all models are combined in this overview.
"""

import argparse
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

# Accept the structure filename from the command line.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("structure_path", help="Path to a PDB mmCIF file")
args = parser.parse_args()

try:
    structure_fields = MMCIF2Dict(args.structure_path)
except (OSError, ValueError) as error:
    parser.exit(1, f"Could not read structure: {error}\n")

# These columns describe the same atoms, so their lengths must match.
required_fields = [
    "_atom_site.label_asym_id",
    "_atom_site.auth_asym_id",
    "_atom_site.label_entity_id",
    "_atom_site.label_comp_id",
]

for field in required_fields:
    if field not in structure_fields:
        parser.error(f"Missing essential field: {field}")

atom_columns = [structure_fields[field] for field in required_fields]
atom_count = len(atom_columns[0])

if atom_count == 0:
    parser.error("The file contains no atom records.")

for field, column in zip(required_fields, atom_columns):
    if len(column) != atom_count:
        parser.error(f"Inconsistent number of atom records in {field}")
    if any(value in {"?", ".", ""} for value in column):
        parser.error(f"Unspecified values in essential field: {field}")

# Entity descriptions are optional. Missing descriptions get a clear label.
entity_ids = structure_fields.get("_entity.id", [])
entity_names = structure_fields.get("_entity.pdbx_description")
descriptions = {}

if entity_names is not None:
    if len(entity_ids) != len(entity_names):
        parser.error("Entity ID and description columns have different lengths.")

    for entity_id, name in zip(entity_ids, entity_names):
        name = " ".join(name.split())
        descriptions[entity_id] = (
            "not provided" if name in {"?", ".", ""} else name
        )

# Group atoms by canonical ID. Sets remove repeated author IDs and codes.
chain_summaries = {}

for chain_id, author_id, entity_id, component_code in zip(*atom_columns):
    if chain_id not in chain_summaries:
        chain_summaries[chain_id] = {
            "author_ids": set(),
            "entity_id": entity_id,
            "components": set(),
        }

    summary = chain_summaries[chain_id]
    if summary["entity_id"] != entity_id:
        parser.error(f"Canonical chain {chain_id} maps to multiple entities.")

    summary["author_ids"].add(author_id)
    summary["components"].add(component_code)

# Print one row per observed canonical chain.
print(f"Structure: {args.structure_path}")
print(f"Atom records: {atom_count}")
print(f"Canonical chains: {len(chain_summaries)}\n")
print(f"{'Canonical ID':<15}{'Author ID':<12}{'Entity ID':<12}Components | Description")
print("-" * 90)

for chain_id, summary in sorted(chain_summaries.items()):
    author_ids = ", ".join(sorted(summary["author_ids"]))
    entity_id = summary["entity_id"]
    components = ", ".join(sorted(summary["components"]))
    description = descriptions.get(entity_id, "not provided")
    print(
        f"{chain_id:<15}{author_ids:<12}{entity_id:<12}"
        f"{components} | {description}"
    )

print("\nShared entity IDs do not establish which protein a ligand binds.")