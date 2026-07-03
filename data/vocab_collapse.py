"""Collapsed vocabularies for the scene-graph generator.

The original MIMIC-Ext-CXR-QBA vocab has 237 entities x 310 regions. Learning
that from noisy B-grade bboxes is essentially impossible: Stage 1 finished at
Val SG Entity Acc = 0.0435 (baseline for 237 classes ~ 0.4%). Per Chest
ImaGenome and the CheXpert labelling scheme, the meaningful anatomical /
finding structure is much smaller, and collapsing to it makes detection
learnable in the compute budget we actually have.

Design choices
--------------
- **Regions**: 29 canonical anatomy classes from Chest ImaGenome plus
  ``other`` = 30.
- **Entities**: 14 CheXpert findings + 6 devices/artifacts + ``normal`` +
  ``other`` = 22.

Both maps are surjective; every original id lands in the collapsed vocab.
Unknown / missing original ids fall through to ``other``.

Usage
-----
::

    from data.vocab_collapse import (
        REGION_COLLAPSE, ENTITY_COLLAPSE,
        NUM_COLLAPSED_REGIONS, NUM_COLLAPSED_ENTITIES,
        collapse_region_id, collapse_entity_id,
    )

    reg_id_collapsed = collapse_region_id(original_id)
    ent_id_collapsed = collapse_entity_id(original_id, name=entity_name)

The collapsed ids are what the SG generator predicts and what the scene-graph
encoder embeds. The dataset loader calls ``collapse_*`` inside
``_extract_gt_targets`` before writing ``gt_entities`` / ``gt_regions`` to
the training batch.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

# =====================================================================
# COLLAPSED REGIONS -- Chest ImaGenome canonical anatomy set (29) + other
# =====================================================================
# Order matters: index in this list is the collapsed region id.
COLLAPSED_REGIONS: List[str] = [
    # Lungs (7)
    "right lung", "left lung", "right upper lung zone", "right mid lung zone",
    "right lower lung zone", "left upper lung zone", "left lower lung zone",
    # Heart / mediastinum (4)
    "cardiac silhouette", "mediastinum", "upper mediastinum", "aortic arch",
    # Pleura (2)
    "right pleural", "left pleural",
    # Costophrenic (2)
    "right costophrenic angle", "left costophrenic angle",
    # Hilum (2)
    "right hilar structures", "left hilar structures",
    # Diaphragm (2)
    "right hemidiaphragm", "left hemidiaphragm",
    # Bones / soft tissue (6)
    "trachea", "spine", "clavicle", "ribs", "abdomen", "neck",
    # Aggregates
    "chest", "cavoatrial junction", "carina",
    # Fallback
    "other",
]

NUM_COLLAPSED_REGIONS = len(COLLAPSED_REGIONS)
_OTHER_REGION_IDX = COLLAPSED_REGIONS.index("other")


# =====================================================================
# COLLAPSED ENTITIES -- 14 CheXpert findings + 6 devices + normal + other
# =====================================================================
COLLAPSED_ENTITIES: List[str] = [
    # CheXpert 14
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "enlarged cardiomediastinum", "fracture", "lung lesion", "lung opacity",
    "no finding", "pleural effusion", "pleural other", "pneumonia",
    "pneumothorax", "support devices",
    # Common devices/artifacts not covered above
    "endotracheal tube", "central venous catheter", "chest tube",
    "nasogastric tube", "pacemaker", "surgical clip",
    # Meta
    "normal",
    # Fallback
    "other",
]

NUM_COLLAPSED_ENTITIES = len(COLLAPSED_ENTITIES)
_OTHER_ENTITY_IDX = COLLAPSED_ENTITIES.index("other")


# =====================================================================
# Name-based collapse: original name (lowercase, stripped) -> new id
# =====================================================================
# The original vocab uses free-text names. We match by substring on the
# lowercased original name. First-match wins; keep specific rules
# BEFORE broad ones (e.g. "left upper lung" must match before "left lung").

_REGION_SUBSTR_RULES: List[tuple] = [
    # Zonal lungs before generic left/right lung
    ("right upper lung", "right upper lung zone"),
    ("right mid lung",   "right mid lung zone"),
    ("right middle lung", "right mid lung zone"),
    ("right lower lung", "right lower lung zone"),
    ("left upper lung",  "left upper lung zone"),
    ("left lower lung",  "left lower lung zone"),
    ("right lung",       "right lung"),
    ("left lung",        "left lung"),
    ("lungs",            "right lung"),  # bilateral -> right (arbitrary)
    ("lung",             "right lung"),
    # Heart / mediastinum
    ("cardiac silhouette", "cardiac silhouette"),
    ("heart",             "cardiac silhouette"),
    ("cardio",            "cardiac silhouette"),
    ("upper mediastinum", "upper mediastinum"),
    ("mediastinum",       "mediastinum"),
    ("aortic arch",       "aortic arch"),
    ("aorta",             "aortic arch"),
    # Pleura
    ("right pleural",     "right pleural"),
    ("left pleural",      "left pleural"),
    ("pleura",            "right pleural"),  # bilateral -> right
    # Costophrenic
    ("right costophrenic", "right costophrenic angle"),
    ("left costophrenic",  "left costophrenic angle"),
    ("costophrenic",       "right costophrenic angle"),
    # Hilum
    ("right hil", "right hilar structures"),
    ("left hil",  "left hilar structures"),
    ("hil",       "right hilar structures"),
    # Diaphragm
    ("right hemidiaphragm", "right hemidiaphragm"),
    ("left hemidiaphragm",  "left hemidiaphragm"),
    ("diaphragm",           "right hemidiaphragm"),
    # Bones / soft tissue
    ("trachea",   "trachea"),
    ("carina",    "carina"),
    ("spine",     "spine"),
    ("clavicle",  "clavicle"),
    ("rib",       "ribs"),
    ("skeleton",  "ribs"),
    ("abdomen",   "abdomen"),
    ("neck",      "neck"),
    ("cavoatrial", "cavoatrial junction"),
    ("chest",     "chest"),
]


_ENTITY_SUBSTR_RULES: List[tuple] = [
    # CheXpert 14 (order sensitive: "pleural effusion" before "pleural other")
    ("atelectasis",     "atelectasis"),
    ("cardiomegaly",    "cardiomegaly"),
    ("consolidation",   "consolidation"),
    ("pulmonary edema", "edema"),
    ("edema",           "edema"),
    ("enlarged cardiomediastinum", "enlarged cardiomediastinum"),
    ("enlarged cardio", "enlarged cardiomediastinum"),
    ("fracture",        "fracture"),
    ("lung lesion",     "lung lesion"),
    ("nodule",          "lung lesion"),
    ("mass",            "lung lesion"),
    ("lung opacity",    "lung opacity"),
    ("opacity",         "lung opacity"),
    ("infiltrat",       "lung opacity"),
    ("no finding",      "no finding"),
    ("pleural effusion", "pleural effusion"),
    ("effusion",        "pleural effusion"),
    ("pleural other",   "pleural other"),
    ("pleural thickening", "pleural other"),
    ("pneumonia",       "pneumonia"),
    ("pneumothorax",    "pneumothorax"),
    # Devices
    ("endotracheal tube", "endotracheal tube"),
    ("et tube",           "endotracheal tube"),
    ("central venous catheter", "central venous catheter"),
    ("cvc",               "central venous catheter"),
    ("picc",              "central venous catheter"),
    ("chest tube",        "chest tube"),
    ("nasogastric",       "nasogastric tube"),
    ("ng tube",           "nasogastric tube"),
    ("pacemaker",         "pacemaker"),
    ("icd",               "pacemaker"),
    ("surgical clip",     "surgical clip"),
    ("clip",              "surgical clip"),
    # Generic support-devices fallback (CheXpert bucket)
    ("device",            "support devices"),
    ("catheter",          "support devices"),
    ("tube",              "support devices"),
    ("line",              "support devices"),
    ("wire",              "support devices"),
    # Meta
    ("normal",            "normal"),
    ("no acute",          "normal"),
    ("intact",            "normal"),
    ("no change",         "normal"),
]


def _build_name_lookup(rules: List[tuple], vocab: List[str]) -> callable:
    """Compile substring rules into a first-match lookup that returns the id."""
    idx_map = {name: i for i, name in enumerate(vocab)}
    for _substr, target in rules:
        assert target in idx_map, (
            f"vocab_collapse rule targets missing vocab entry: {target!r}"
        )
    def _lookup(name: Optional[str], default_idx: int) -> int:
        if not name:
            return default_idx
        n = str(name).strip().lower()
        for substr, target in rules:
            if substr in n:
                return idx_map[target]
        return default_idx
    return _lookup


_lookup_region_by_name = _build_name_lookup(_REGION_SUBSTR_RULES, COLLAPSED_REGIONS)
_lookup_entity_by_name = _build_name_lookup(_ENTITY_SUBSTR_RULES, COLLAPSED_ENTITIES)


# =====================================================================
# Public API
# =====================================================================
def collapse_region_id(
    original_id: Optional[int],
    *,
    name: Optional[str] = None,
    original_vocab: Optional[List[str]] = None,
) -> int:
    """Map an original region id (or name) to a collapsed region id.

    ``original_vocab`` lets you pass the loaded ``region_names`` list from
    ``dataset_info.json`` so the collapse can consult the original name.
    If both ``original_id`` and ``name`` are given, ``name`` wins.
    """
    if not name and original_id is not None and original_vocab is not None:
        if 0 <= int(original_id) < len(original_vocab):
            name = original_vocab[int(original_id)]
    return _lookup_region_by_name(name, _OTHER_REGION_IDX)


def collapse_entity_id(
    original_id: Optional[int],
    *,
    name: Optional[str] = None,
    original_vocab: Optional[List[str]] = None,
) -> int:
    """Map an original entity id (or name) to a collapsed entity id."""
    if not name and original_id is not None and original_vocab is not None:
        if 0 <= int(original_id) < len(original_vocab):
            name = original_vocab[int(original_id)]
    return _lookup_entity_by_name(name, _OTHER_ENTITY_IDX)


def collapse_region_ids_bulk(
    original_ids: Iterable[int],
    original_vocab: Optional[List[str]] = None,
) -> List[int]:
    return [
        collapse_region_id(int(i), original_vocab=original_vocab)
        for i in original_ids
    ]


def collapse_entity_ids_bulk(
    original_ids: Iterable[int],
    original_vocab: Optional[List[str]] = None,
) -> List[int]:
    return [
        collapse_entity_id(int(i), original_vocab=original_vocab)
        for i in original_ids
    ]


# Exposed as immutable dicts so callers can also introspect the collapse
# table without going through the lookup function.
REGION_COLLAPSE: Dict[str, str] = {rule[0]: rule[1] for rule in _REGION_SUBSTR_RULES}
ENTITY_COLLAPSE: Dict[str, str] = {rule[0]: rule[1] for rule in _ENTITY_SUBSTR_RULES}
