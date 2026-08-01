"""
hardware_mapping.py

Single source of truth for the fNIRS hardware channel mapping.

WHAT COMES FROM WHERE (read this first)
-----------------------------------------------------------------------
- Component POSITIONS (pcb_x_mm, pcb_y_mm, rotation, board side) come
  from the KiCad PCB file. These are extracted, not configured.

- Component IDENTITY (is this designator an LED or a photodetector) is
  VERIFIED against the PCB: every LED designator below must resolve to a
  footprint with Value="LED" / Footprint="LED_SMD:LED_1206_3216Metric",
  and every detector designator must resolve to Value="VBPW34S" /
  Footprint="OptoDevice:Osram_BPW34S-SMD". geometry/pcb_geometry.py
  builds pcb_led_refs / pcb_detector_refs directly from the PCB by
  matching on (value, footprint) and cross-checks them against the sets
  below - it does not just trust that the listed designators exist.

- LED WAVELENGTH ASSIGNMENT (which 8 of the 16 LEDs are 740nm vs 850nm)
  is NOT extractable from the PCB at all: both wavelengths use the
  identical generic footprint and Value="LED". Which physical LED part
  (740nm vs 850nm) got placed in which footprint is an assembly/BOM
  decision with no schematic-level trace. THIS ASSIGNMENT IS THEREFORE
  AN EXPLICIT, EXTERNALLY-SUPPLIED CONFIGURATION, not something this
  code discovers. If it's ever wrong, it has to be corrected here, by
  a human who knows what was actually assembled.

- HARDWARE ACQUISITION ORDER (the order the LED driver fires sources in,
  the ADC/mux channel a detector is wired to, etc.) is UNKNOWN from the
  supplied schematic and firmware. We do not invent it. The optional
  index fields below are left as -1 (see channel_map.py) and the
  validation report says explicitly: "The acquisition order could not
  be verified from the supplied files."
-----------------------------------------------------------------------
"""

from __future__ import annotations

MAPPING_VERSION = "2.0.0"

# --- Explicit, externally-supplied wavelength assignment -----------------
LED_740_ORDER: list[str] = ["D2", "D3", "D5", "D6", "D25", "D26", "D29", "D30"]
LED_850_ORDER: list[str] = ["D1", "D4", "D7", "D8", "D27", "D28", "D31", "D32"]

# Backward-compatible aliases (some earlier scripts/prompt text use these names)
LEDS_740 = LED_740_ORDER
LEDS_850 = LED_850_ORDER

# --- Fixed, explicit detector / matrix-column order -----------------------
# This determines matrix_column for every sample. Must never come from a
# Python dict/set (insertion-order/hash-order is not a stable contract),
# alphabetical sort, or PCB/KiCad internal component order.
DETECTOR_ORDER: list[str] = [
    "D9", "D10", "D11", "D12",
    "D13", "D14", "D15", "D16",
    "D17", "D18", "D19", "D20",
    "D21", "D22", "D23", "D24",
]
DETECTORS = DETECTOR_ORDER  # alias

# --- Expected PCB component identity (used to VERIFY, not assume) --------
LED_VALUE = "LED"
LED_FOOTPRINT = "LED_SMD:LED_1206_3216Metric"

DETECTOR_VALUE = "VBPW34S"
DETECTOR_FOOTPRINT = "OptoDevice:Osram_BPW34S-SMD"

# --- Hardware acquisition order: NOT AVAILABLE -----------------------------
# These would map matrix_row/matrix_column to the physical LED-driver /
# ADC / multiplexer channel index. The supplied schematic and firmware do
# not provide enough information to fill these in reliably, so they are
# fixed at -1 (meaning "unknown / not verified") everywhere. Do not set
# these to anything other than -1 without a documented source (schematic
# net name, firmware channel table, etc.).
LED_DRIVE_INDEX_KNOWN = False
DETECTOR_ADC_INDEX_KNOWN = False
HARDWARE_CHANNEL_INDEX_KNOWN = False
UNKNOWN_INDEX = -1
