"""PMCX optical runner plus an explicitly non-physical analytic_debug backend for CI only."""
from __future__ import annotations

import time
import numpy as np

from configs.simulation_config import PMCXConfig
from simulation.detector_model import extract_detector_values


class PMCXUnavailableError(RuntimeError):
    pass


class OpticalSimulationError(RuntimeError):
    pass


_EXPLICIT_DETECTOR_MODES = {
    "explicit",
    "explicit_detected_photons",
    "detected_photons",
    "mcx_detectors",
}


def _validate_source(
    volume: np.ndarray,
    source_mm: np.ndarray,
    voxel_size_mm: float,
    srcdir,
) -> np.ndarray:
    p = np.asarray(source_mm, dtype=float) / voxel_size_mm
    d = np.asarray(srcdir, dtype=float)
    n = np.linalg.norm(d)

    if not np.isfinite(p).all() or not np.isfinite(d).all() or n == 0:
        raise OpticalSimulationError("Invalid source position or direction")

    d = d / n
    idx = np.rint(p).astype(int)

    if np.any(idx < 0) or np.any(idx >= np.array(volume.shape)):
        raise OpticalSimulationError(
            f"Source outside volume: {np.asarray(source_mm).tolist()} mm"
        )

    if volume[tuple(idx)] == 0:
        raise OpticalSimulationError(
            f"Source is in air at voxel {idx.tolist()}; "
            "expected first scalp voxel"
        )

    step = np.rint(p + d).astype(int)
    if (
        np.any(step < 0)
        or np.any(step >= np.array(volume.shape))
        or volume[tuple(step)] == 0
    ):
        raise OpticalSimulationError(
            "Source direction does not point into tissue"
        )

    return d


def _safe_pmcx_seed(seed: int) -> int:
    value = int(seed) % 2_147_483_647
    return value if value > 0 else 1


def _build_explicit_detpos(
    detectors_mm: np.ndarray,
    detector_radius_mm: float,
    voxel_size_mm: float,
) -> np.ndarray:
    detectors_mm = np.asarray(detectors_mm, dtype=np.float32)

    if detectors_mm.ndim != 2 or detectors_mm.shape[1] != 3:
        raise OpticalSimulationError(
            f"Expected detector positions with shape (N,3), got {detectors_mm.shape}"
        )

    detector_positions_vox = detectors_mm / float(voxel_size_mm)
    detector_radius_vox = float(detector_radius_mm) / float(voxel_size_mm)

    return np.column_stack(
        [
            detector_positions_vox,
            np.full(
                detector_positions_vox.shape[0],
                detector_radius_vox,
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def _extract_explicit_detector_values(
    result: dict,
    props: np.ndarray,
    detector_count: int,
    voxel_size_mm: float,
    photon_count: int,
    savedetflag: str,
) -> np.ndarray:
    try:
        from pmcx.utils import detphoton, detweight
    except ImportError:
        import pmcx
        detphoton = pmcx.detphoton
        detweight = pmcx.detweight

    if not isinstance(result, dict) or "detp" not in result:
        keys = list(result) if isinstance(result, dict) else type(result)
        raise OpticalSimulationError(f"PMCX returned no detp field; keys={keys}")

    raw_detp = np.asarray(result["detp"])
    medium_count = int(np.asarray(props).shape[0] - 1)

    try:
        parsed = detphoton(raw_detp, medium_count, savedetflag)
        weights = np.asarray(
            detweight(
                parsed,
                np.asarray(props, dtype=np.float32),
                unitinmm=float(voxel_size_mm),
            ),
            dtype=np.float64,
        ).reshape(-1)
    except Exception as exc:
        raise OpticalSimulationError(
            f"Failed to parse/weight PMCX detected photons: {exc}"
        ) from exc

    detector_ids = np.asarray(parsed["detid"], dtype=np.int64).reshape(-1)

    if detector_ids.size != weights.size:
        raise OpticalSimulationError(
            "detid and detected-photon weight lengths do not match"
        )

    if detector_ids.size == 0:
        return np.zeros(detector_count, dtype=np.float64)

    if np.any((detector_ids < 1) | (detector_ids > detector_count)):
        raise OpticalSimulationError(
            "PMCX returned detector IDs outside the expected range"
        )

    values = np.bincount(
        detector_ids,
        weights=weights,
        minlength=detector_count + 1,
    )[1 : detector_count + 1]

    return np.asarray(values / float(photon_count), dtype=np.float64)


def _run_pmcx(
    volume,
    source_mm,
    detectors_mm,
    props,
    voxel_size_mm,
    cfg,
    seed,
):
    try:
        import pmcx
    except ImportError as exc:
        raise PMCXUnavailableError(
            "pmcx is not installed. Install PMCX/CUDA and rerun with --backend pmcx."
        ) from exc

    direction = _validate_source(
        volume,
        source_mm,
        voxel_size_mm,
        cfg.srcdir,
    )

    source_vox = np.asarray(source_mm, dtype=np.float32) / float(voxel_size_mm)
    detector_mode = str(cfg.detector_mode).strip().lower()
    use_explicit_detectors = detector_mode in _EXPLICIT_DETECTOR_MODES

    mcx_cfg = {
        "nphoton": int(cfg.photon_count),
        "vol": np.asarray(volume, dtype=np.uint8),
        "srcpos": source_vox,
        "srcdir": direction.astype(np.float32),
        "prop": np.asarray(props, dtype=np.float32),
        "tstart": float(cfg.tstart),
        "tstep": float(cfg.tstep),
        "tend": float(cfg.tend),
        "issrcfrom0": int(cfg.issrcfrom0),
        "seed": _safe_pmcx_seed(seed),
        "unitinmm": float(voxel_size_mm),
    }

    savedetflag = "dpxv"

    if use_explicit_detectors:
        mcx_cfg.update(
            {
                "detpos": _build_explicit_detpos(
                    detectors_mm,
                    cfg.detector_radius_mm,
                    voxel_size_mm,
                ),
                "issavedet": 1,
                "savedetflag": savedetflag,
                "maxdetphoton": int(cfg.photon_count),
            }
        )

    started = time.perf_counter()

    try:
        result = pmcx.run(mcx_cfg)
    except Exception as exc:
        raise OpticalSimulationError(f"PMCX failed: {exc}") from exc

    if use_explicit_detectors:
        values = _extract_explicit_detector_values(
            result=result,
            props=np.asarray(props, dtype=np.float32),
            detector_count=len(detectors_mm),
            voxel_size_mm=voxel_size_mm,
            photon_count=int(cfg.photon_count),
            savedetflag=savedetflag,
        )
        return values, time.perf_counter() - started

    field = result.get("flux") if isinstance(result, dict) else None
    if field is None and isinstance(result, dict):
        field = result.get("data")
    if field is None:
        raise OpticalSimulationError(
            f"PMCX returned no flux/data field; "
            f"keys={list(result) if isinstance(result, dict) else type(result)}"
        )

    field = np.asarray(field)
    if field.ndim == 4:
        field = field.sum(axis=3)

    if field.shape != volume.shape:
        raise OpticalSimulationError(
            f"Unexpected PMCX field shape {field.shape}, expected {volume.shape}"
        )

    values = extract_detector_values(
        field,
        detectors_mm,
        voxel_size_mm,
        cfg.detector_mode,
        cfg.detector_radius_mm,
    )
    return values, time.perf_counter() - started


def _run_analytic_debug(
    volume,
    source_mm,
    detectors_mm,
    props,
    voxel_size_mm,
    cfg,
    seed,
):
    """Fast deterministic smoke-test backend. NOT MCX and NOT valid training data."""
    _validate_source(volume, source_mm, voxel_size_mm, cfg.srcdir)

    source = np.asarray(source_mm, float)
    detectors = np.asarray(detectors_mm, float)
    distances = np.linalg.norm(detectors - source[None, :], axis=1)

    values = []
    for detector, distance in zip(detectors, distances):
        sample_count = max(10, int(np.ceil(distance / voxel_size_mm)) * 2)
        points = np.linspace(source, detector, sample_count)
        indices = np.rint(points / voxel_size_mm).astype(int)
        indices = np.clip(indices, [0, 0, 0], np.array(volume.shape) - 1)

        labels = volume[
            indices[:, 0],
            indices[:, 1],
            indices[:, 2],
        ]
        mua = float(np.mean(props[labels, 0]))
        mus = float(np.mean(props[labels, 1]))

        values.append(
            np.exp(-(mua + 0.01 * mus) * max(distance, 1.0))
            / (max(distance, 1.0) ** 2)
        )

    return np.asarray(values, float), 0.0


def run_source(
    volume,
    source_mm,
    detectors_mm,
    props,
    voxel_size_mm,
    cfg: PMCXConfig,
    seed: int,
):
    if cfg.backend == "pmcx":
        return _run_pmcx(
            volume,
            source_mm,
            detectors_mm,
            props,
            voxel_size_mm,
            cfg,
            seed,
        )
    if cfg.backend == "analytic_debug":
        return _run_analytic_debug(
            volume,
            source_mm,
            detectors_mm,
            props,
            voxel_size_mm,
            cfg,
            seed,
        )
    raise ValueError(f"Unknown backend {cfg.backend!r}")


def run_wavelength(
    volume,
    sources_mm,
    detectors_mm,
    props,
    voxel_size_mm,
    cfg,
    base_seed,
    progress_callback=None,
):
    if len(sources_mm) != 8 or len(detectors_mm) != 16:
        raise AssertionError(
            "Hardware requires 8 sources and 16 detectors"
        )

    matrix = np.zeros((8, 16), dtype=np.float64)
    times = []

    for source_index, source in enumerate(sources_mm):
        values, elapsed = run_source(
            volume,
            source,
            detectors_mm,
            props,
            voxel_size_mm,
            cfg,
            base_seed + source_index,
        )
        matrix[source_index] = values
        times.append(elapsed)

        if progress_callback:
            progress_callback(source_index, elapsed)

    validate_measurement_matrix(matrix)
    return matrix, np.asarray(times)


def validate_measurement_matrix(matrix):
    if matrix.shape != (8, 16):
        raise OpticalSimulationError(
            f"Expected (8,16), got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise OpticalSimulationError(
            "Measurement contains NaN/Inf"
        )
    if (matrix < 0).any():
        raise OpticalSimulationError(
            "Measurement contains negative values"
        )
    if np.any(np.all(matrix == 0, axis=1)):
        raise OpticalSimulationError(
            "At least one complete source row is zero"
        )


def matrix_stats(matrix):
    positive = matrix[matrix > 0]
    return {
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "mean": float(matrix.mean()),
        "median": float(np.median(matrix)),
        "std": float(matrix.std()),
        "zero_count": int((matrix == 0).sum()),
        "negative_count": int((matrix < 0).sum()),
        "nan_count": int(np.isnan(matrix).sum()),
        "inf_count": int(np.isinf(matrix).sum()),
        "min_positive": (
            float(positive.min())
            if positive.size
            else float("nan")
        ),
    }
