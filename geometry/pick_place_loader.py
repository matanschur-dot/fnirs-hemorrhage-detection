"""
pick_place_loader.py

Optional FALLBACK geometry source. The KiCad PCB parser (kicad_parser.py)
remains the primary and preferred geometry source and should be used
whenever a .kicad_pcb file is available - it lets us verify Value AND
Footprint against the real component library, which a plain position
export usually can't. This loader exists for the case where only a
pick-and-place / centroid CSV is available (e.g. from a PCB fab, without
the original KiCad project).

Isolated on purpose: this module produces the exact same PCBComponent
shape that kicad_parser.py does, so everything downstream (pcb_geometry,
channel_map, validation) works unchanged regardless of which loader was
used. It does NOT reimplement hardware validation or channel mapping.

LIMITATION: most pick-and-place exports do not carry KiCad's internal
`Value`/library `Footprint` fields verbatim - they carry a Value/Comment
column and a Footprint/Package column. This loader maps common column
name variants but does not try to be clever about it; if your CSV's
values/footprints don't match EXPECTED_LED_FOOTPRINT/EXPECTED_DETECTOR_FOOTPRINT
in configs/hardware_mapping.py exactly, hardware validation will
(correctly) fail rather than silently passing on a fuzzy match. Adjust
those constants, or the CSV, so they agree - don't loosen the validation
to make a mismatch pass.
"""

from __future__ import annotations

import csv

from geometry.kicad_parser import PCBComponent

_REF_COLS = ["Ref", "Designator", "Reference", "RefDes"]
_X_COLS = ["Mid X", "PosX", "X", "Center-X(mm)", "X (mm)"]
_Y_COLS = ["Mid Y", "PosY", "Y", "Center-Y(mm)", "Y (mm)"]
_ROT_COLS = ["Rotation", "Rot", "Rotation (deg)"]
_SIDE_COLS = ["Side", "Layer", "TB"]
_VALUE_COLS = ["Val", "Value", "Comment"]
_FOOTPRINT_COLS = ["Footprint", "Package", "PackageRef"]


def _first_present(row: dict, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return row[c]
    return None


def load_positions_from_csv(csv_path: str) -> dict[str, PCBComponent]:
    """Parse a pick-and-place / centroid CSV into the same PCBComponent
    shape kicad_parser.py produces, so it's a drop-in alternative to
    `components_by_reference()` for pcb_geometry.build_channel_map() etc."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in pick-and-place CSV: {csv_path}")

    components: dict[str, PCBComponent] = {}
    for row in rows:
        ref = _first_present(row, _REF_COLS)
        x = _first_present(row, _X_COLS)
        y = _first_present(row, _Y_COLS)
        if ref is None or x is None or y is None:
            raise ValueError(
                f"Row missing reference/X/Y in {csv_path}: {row}. "
                f"Expected one of {_REF_COLS} / {_X_COLS} / {_Y_COLS}."
            )
        rot = _first_present(row, _ROT_COLS)
        side = _first_present(row, _SIDE_COLS) or "UNKNOWN"
        value = _first_present(row, _VALUE_COLS) or ""
        footprint = _first_present(row, _FOOTPRINT_COLS) or "UNKNOWN"

        ref = ref.strip()
        if ref in components:
            raise ValueError(f"Duplicate reference designator '{ref}' found in {csv_path}.")

        components[ref] = PCBComponent(
            reference=ref, value=value, footprint=footprint,
            pcb_x_mm=float(x), pcb_y_mm=float(y),
            rotation_deg=float(rot) if rot is not None else 0.0,
            board_side=side,
        )

    return components
