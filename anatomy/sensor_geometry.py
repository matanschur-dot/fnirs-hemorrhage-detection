"""Convert verified PCB x-y coordinates to flat-phantom optical positions."""
from __future__ import annotations
import numpy as np
from geometry.kicad_parser import PCBComponent
from geometry.pcb_geometry import compute_transform, pcb_to_head
from configs.hardware_mapping import LED_740_ORDER, LED_850_ORDER, DETECTOR_ORDER
from configs.geometry_config import GeometryRunConfig
from configs.anatomy_config import SourceDetectorPlacementConfig
from anatomy.subject_generator import SensorGeometry

def build_sensor_geometry(components: dict[str, PCBComponent], geometry_cfg: GeometryRunConfig,
                          placement_cfg: SourceDetectorPlacementConfig,
                          tissue_surface_z_mm: float | None = None) -> SensorGeometry:
    transform = compute_transform(components, geometry_cfg)
    # Before anatomy exists, retain legacy candidate z. For optical/anatomy generation,
    # callers pass the actual air/scalp boundary.
    surface = geometry_cfg.surface.surface_z_mm if tissue_surface_z_mm is None else tissue_surface_z_mm
    def positions(refs: list[str], z_mm: float) -> np.ndarray:
        out = np.zeros((len(refs), 3), dtype=np.float32)
        for i, ref in enumerate(refs):
            c = components[ref]
            hx, hy = pcb_to_head(c.pcb_x_mm, c.pcb_y_mm, transform)
            out[i] = (hx, hy, z_mm)
        return out
    return SensorGeometry(
        source_positions_740_mm=positions(LED_740_ORDER, surface + placement_cfg.source_offset_into_scalp_mm),
        source_positions_850_mm=positions(LED_850_ORDER, surface + placement_cfg.source_offset_into_scalp_mm),
        detector_positions_mm=positions(DETECTOR_ORDER, surface + placement_cfg.detector_offset_into_scalp_mm),
    )
