"""
kicad_parser.py

Minimal, dependency-free parser for KiCad 6/7/8 .kicad_pcb files.

KiCad PCB files are S-expressions. Rather than pulling in a general
S-expression grammar library (more machinery than we need, and another
place for subtle bugs), we scan for `(footprint ...)` blocks using a
paren-depth counter to find each block's exact extent, then regex the
specific fields we need out of that substring. This is intentionally
narrow in scope and raises loudly if an expected field is missing,
rather than silently returning incomplete/wrong data.

Extracted per footprint:
  - reference   e.g. "D9"           (property "Reference")
  - value       e.g. "VBPW34S"       (property "Value")
  - footprint   e.g. "OptoDevice:Osram_BPW34S-SMD"   (the lib_id right
                after the opening `(footprint "..."`)
  - pcb_x_mm, pcb_y_mm, rotation_deg   from `(at x y [rotation])`
  - board_side  the copper layer the footprint is placed on, taken from
                the footprint's own `(layer "...")` field (e.g. "F.Cu"
                or "B.Cu")

If direct PCB parsing ever breaks (e.g. a KiCad version bump changes the
file format), fall back to a position-export CSV - not implemented here
since none was supplied for this revision, but the PCBComponent shape
below is what such a fallback loader would need to produce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PCBComponent:
    reference: str        # e.g. "D9"
    value: str             # e.g. "VBPW34S" or "LED"
    footprint: str          # lib_id, e.g. "OptoDevice:Osram_BPW34S-SMD"
    pcb_x_mm: float
    pcb_y_mm: float
    rotation_deg: float
    board_side: str          # raw layer string, e.g. "F.Cu" / "B.Cu"


def _find_footprint_blocks(text: str) -> list[str]:
    """Return the raw text of every (footprint ...) block, matched by
    paren depth so nested parens inside the block don't confuse us."""
    blocks = []
    start_token = "(footprint "
    idx = 0
    while True:
        start = text.find(start_token, idx)
        if start == -1:
            break
        depth = 0
        i = start
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            raise ValueError(
                f"Unbalanced parentheses while scanning footprint block "
                f"starting at char {start}. File may be truncated or corrupt."
            )
        blocks.append(text[start:i + 1])
        idx = i + 1
    return blocks


def _extract_field(block: str, pattern: str, field_name: str, required: bool = True) -> str | None:
    m = re.search(pattern, block)
    if m is None:
        if required:
            raise ValueError(f"Could not find required field '{field_name}' in footprint block.")
        return None
    return m.group(1)


def parse_footprints(pcb_path: str) -> list[PCBComponent]:
    """Parse a .kicad_pcb file and return every footprint found."""
    with open(pcb_path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = _find_footprint_blocks(text)
    if not blocks:
        raise ValueError(
            f"No '(footprint ...)' blocks found in {pcb_path}. "
            f"Is this a valid KiCad PCB file?"
        )

    components: list[PCBComponent] = []
    for block in blocks:
        lib_id_match = re.match(r'\(footprint\s+"([^"]+)"', block)
        footprint_lib = lib_id_match.group(1) if lib_id_match else "UNKNOWN"

        ref = _extract_field(block, r'\(property\s+"Reference"\s+"([^"]+)"', "Reference")
        value = _extract_field(block, r'\(property\s+"Value"\s+"([^"]+)"', "Value", required=False) or ""

        # Footprint's own placement layer, e.g. (layer "F.Cu"), appears
        # right after the lib_id/locked/placed tokens, before any nested
        # (pad ...) blocks that also contain (layer ...). Take the FIRST
        # top-level occurrence.
        layer_match = re.search(r'\(layer\s+"([^"]+)"\)', block)
        board_side = layer_match.group(1) if layer_match else "UNKNOWN"

        at_match = re.search(r'\(at\s+([\-\d.]+)\s+([\-\d.]+)(?:\s+([\-\d.]+))?\)', block)
        if at_match is None:
            raise ValueError(f"Footprint '{ref}' has no (at x y [rotation]) position field.")
        x_mm = float(at_match.group(1))
        y_mm = float(at_match.group(2))
        rotation_deg = float(at_match.group(3)) if at_match.group(3) is not None else 0.0

        components.append(PCBComponent(
            reference=ref, value=value, footprint=footprint_lib,
            pcb_x_mm=x_mm, pcb_y_mm=y_mm, rotation_deg=rotation_deg,
            board_side=board_side,
        ))

    return components


def components_by_reference(pcb_path: str) -> dict[str, PCBComponent]:
    comps = parse_footprints(pcb_path)
    by_ref: dict[str, PCBComponent] = {}
    for c in comps:
        if c.reference in by_ref:
            raise ValueError(
                f"Duplicate reference designator '{c.reference}' found in {pcb_path}. "
                f"Refusing to continue with an ambiguous PCB file."
            )
        by_ref[c.reference] = c
    return by_ref


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "fnirs.kicad_pcb"
    comps = components_by_reference(path)
    diode_refs = sorted(
        (r for r in comps if re.match(r"^D\d+$", r)),
        key=lambda r: int(r[1:]),
    )
    print(f"Found {len(comps)} total footprints, {len(diode_refs)} diode-family (D*) designators.\n")
    for r in diode_refs:
        c = comps[r]
        print(f"{r:>5s}  footprint={c.footprint:<32s} value={c.value:<10s} "
              f"x={c.pcb_x_mm:8.3f}  y={c.pcb_y_mm:8.3f}  rot={c.rotation_deg:6.1f}  side={c.board_side}")
