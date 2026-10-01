import json
from pathlib import Path

from chimerax.atomic import AtomicStructure
from chimerax.core.commands import run

root = Path(__file__).resolve().parent
pairs = json.loads((root / "pairs.json").read_text())

if not pairs:
    raise RuntimeError("No eligible designs in this bundle. Read summary.json.")
if session.models.list():
    raise RuntimeError("Open this viewer in an empty ChimeraX session.")

def command(text):
    return run(session, text)

def load(relative):
    existing = set(session.models.list())
    path = str(root / relative)
    command("open " + json.dumps(path))
    added = [
        m for m in session.models.list()
        if m not in existing and isinstance(m, AtomicStructure)
    ]
    if len(added) != 1:
        raise RuntimeError(f"Expected one atomic model in {relative}")
    return added[0]

def identify(model):
    # Conservative rules for this MurC + short peptide project.
    chains = []
    for chain in model.chains:
        residues = [r for r in chain.residues if r is not None]
        protein = [r for r in residues if r.find_atom("CA") is not None]
        if protein:
            chains.append((chain, len(protein)))
    targets = [c for c, n in chains if n >= 200]
    peptides = [c for c, n in chains if 2 <= n <= 100]
    if len(targets) != 1 or len(peptides) != 1 or len(chains) != 2:
        raise RuntimeError(
            f"Ambiguous chains in {model.name}: "
            + str([(c.chain_id, n) for c, n in chains])
        )
    return targets[0], peptides[0]

def chain_spec(model, chain):
    return f"#{model.id_string}/{chain.chain_id}"

command("set bgColor white")
command("lighting soft")

# One reference receptor for a consistent orientation across all pairs.
reference = load(pairs[0]["before"])
reference.name = "MurC reference (first pre-refold structure)"
ref_target, ref_peptide = identify(reference)
ref_spec = chain_spec(reference, ref_target)
command(f"hide #{reference.id_string} atoms")
command(f"hide #{reference.id_string} cartoons")
command(f"show {ref_spec} cartoons")
command(f"color {ref_spec} lightgray")

groups = []
for item in pairs:
    before = load(item["before"])
    after = load(item["after"])
    bt, bp = identify(before)
    at, ap = identify(after)

    before.name = "Before refolding — cyan"
    after.name = "After refolding — magenta"

    for model, target, peptide, colour in [
        (before, bt, bp, "cyan"),
        (after, at, ap, "magenta"),
    ]:
        target_spec = chain_spec(model, target)
        peptide_spec = chain_spec(model, peptide)
        # Match the receptor; move the entire complex together.
        command(f"matchmaker {target_spec} to {ref_spec}")
        command(f"hide #{model.id_string} atoms")
        command(f"hide #{model.id_string} cartoons")
        command(f"show {peptide_spec} cartoons")
        command(f"color {peptide_spec} {colour}")

    command(
        f"transparency {chain_spec(before, bp)} 50 target c"
    )
    command(f"show {chain_spec(after, ap)} atoms")
    command(f"style {chain_spec(after, ap)} stick")

    group_id = max(
        m.id[0] for m in session.models.list() if m.id
    ) + 1
    command(
        f"rename #{before.id_string},{after.id_string} "
        f"id #{group_id}"
    )
    group = next(
        m for m in session.models.list()
        if m.id == (group_id,)
    )
    group.name = item["design"] + " [pass_filters=" + str(item.get("pass_filters", "unknown")) + "]"
    groups.append(group)

    session.logger.info(
        f"{item['design']}: "
        f"before MurC={bt.chain_id}, peptide={bp.chain_id}; "
        f"after MurC={at.chain_id}, peptide={ap.chain_id}"
    )

for index, group in enumerate(groups):
    group.display = (index == 0)

command("view")
session.logger.info(
    f"Loaded {len(groups)} design pairs. "
    "Use the Model Panel to switch groups: cyan = before, "
    "magenta = after. Grey MurC is a common reference, "
    "not each design's own receptor conformation."
)
