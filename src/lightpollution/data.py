"""Read geographical windows of VNP46A3/A4 Collection 2 and audit their attributes."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from .download import sha256
from .periods import CompositePeriod

PIXELS_PER_DEGREE = 240
GRID = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
COMPOSITES = ("NearNadir", "OffNadir", "AllAngle")
RADIANCE_UNIT = "nW/(cm^2 sr)"


@dataclass(frozen=True)
class Bounds:
    west: float = 29.0
    south: float = 59.0
    east: float = 32.0
    north: float = 61.0

    def __post_init__(self):
        if not (-180 <= self.west < self.east <= 180 and -90 <= self.south < self.north <= 90):
            raise ValueError("Bounds must be ordered and must not cross the antimeridian.")
        for value in asdict(self).values():
            if not np.isclose(value * PIXELS_PER_DEGREE, round(value * PIXELS_PER_DEGREE), atol=1e-7, rtol=0):
                raise ValueError("Bounds must align with the 15 arc-second grid.")

    @property
    def shape(self) -> tuple[int, int]:
        return (round((self.north - self.south) * PIXELS_PER_DEGREE),
                round((self.east - self.west) * PIXELS_PER_DEGREE))

    @property
    def latitude(self) -> np.ndarray:
        return self.north - (np.arange(self.shape[0]) + 0.5) / PIXELS_PER_DEGREE

    @property
    def longitude(self) -> np.ndarray:
        return self.west + (np.arange(self.shape[1]) + 0.5) / PIXELS_PER_DEGREE

    def tiles(self) -> tuple[str, ...]:
        first_h = int(np.floor((self.west + 180) / 10))
        last_h = int(np.ceil((self.east + 180) / 10))
        first_v = int(np.floor((90 - self.north) / 10))
        last_v = int(np.ceil((90 - self.south) / 10))
        return tuple(f"h{h:02}v{v:02}" for v in range(first_v, last_v) for h in range(first_h, last_h))


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value,(list,tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    return str(value)


def text_attribute(attributes, name: str) -> str:
    value = json_value(attributes[name])
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    return str(value).strip()


def decode(dataset: h5py.Dataset, window: tuple[slice, slice]) -> np.ndarray:
    """Mask raw fill values before applying scale/offset. No C1 defaults."""
    raw = np.asarray(dataset[window])
    if "_FillValue" not in dataset.attrs or "scale_factor" not in dataset.attrs:
        raise ValueError(f"Required fill/scale attributes missing for {dataset.name}.")
    fill = np.asarray(dataset.attrs["_FillValue"]).item()
    scale = float(np.asarray(dataset.attrs["scale_factor"]).item())
    if "add_offset" in dataset.attrs:
        offset = float(np.asarray(dataset.attrs["add_offset"]).item())
    elif "offset" in dataset.attrs:
        offset = float(np.asarray(dataset.attrs["offset"]).item())
    else:
        raise ValueError(f"Offset attribute missing for {dataset.name}.")
    # VNP46A4.002 uses scale=1, offset=0. Fail on an undocumented non-zero
    # offset rather than silently choosing between CF and HDF-EOS conventions.
    if offset != 0:
        raise ValueError(f"Non-zero Black Marble offset needs an explicit convention: {dataset.name}.")
    result = raw.astype(np.float64) * scale
    result[(raw == fill) | ~np.isfinite(raw)] = np.nan
    return result


def good_mask(radiance: np.ndarray, quality: np.ndarray, observations: np.ndarray) -> np.ndarray:
    # NASA User Guide C2, table 12: good quality requires MORE than 3 observations.
    return np.isfinite(radiance) & (radiance >= 0) & (quality == 0) & (observations > 3)


def read_region(directory: Path, bounds: Bounds, year: int = 2025,
                month: int | None = None) -> tuple[dict, dict]:
    period = CompositePeriod(year, month)
    arrays: dict[str, np.ndarray] = {}
    attributes: dict = {}
    coverage = np.zeros(bounds.shape, dtype=np.uint8)
    for tile in bounds.tiles():
        files = sorted(directory.glob(f"{period.product}.A{year:04d}{period.day_of_year}.{tile}.002.*.h5"))
        if len(files) != 1:
            raise ValueError(f"Expected one pinned file for {tile}; found {len(files)}.")
        path = files[0]
        match = re.fullmatch(r"h(\d{2})v(\d{2})", tile)
        h, v = (int(x) for x in match.groups())
        west, north = -180 + 10*h, 90 - 10*v
        left, right = max(west, bounds.west), min(west+10, bounds.east)
        bottom, top = max(north-10, bounds.south), min(north, bounds.north)
        src = (slice(round((north-top)*240), round((north-bottom)*240)),
               slice(round((left-west)*240), round((right-west)*240)))
        dst = (slice(round((bounds.north-top)*240), round((bounds.north-bottom)*240)),
               slice(round((left-bounds.west)*240), round((right-bounds.west)*240)))
        with h5py.File(path, "r") as handle:
            if text_attribute(handle.attrs, "ShortName") != period.product or text_attribute(handle.attrs, "VersionID") != "002":
                raise ValueError(f"Wrong product or collection in {path.name}.")
            if (text_attribute(handle.attrs, "RangeBeginningDate") != period.start.isoformat()
                    or text_attribute(handle.attrs, "RangeEndingDate") != period.end.isoformat()):
                raise ValueError(f"Unexpected observation period in {path.name}.")
            for name, expected in (("WestBoundingCoord", west), ("EastBoundingCoord", west+10),
                                   ("NorthBoundingCoord", north), ("SouthBoundingCoord", north-10)):
                if not np.isclose(float(np.asarray(handle.attrs[name]).item()), expected, atol=1e-6):
                    raise ValueError(f"Geographic metadata mismatch: {path.name}, {name}.")
            fields = handle[GRID]
            tile_info = {
                "filename": path.name,
                "sha256": sha256(path),
                "global_attributes": {key: json_value(value) for key, value in handle.attrs.items() if key != "InputPointer"},
                "datasets": {},
                "coordinate_conventions": {},
            }
            names = [f"{prefix}_Composite_Snow_Free{suffix}" for prefix in COMPOSITES
                     for suffix in ("", "_Quality", "_Num", "_Std")] + ["Land_Water_Mask"]
            for name in names:
                dataset = fields[name]
                if dataset.shape != (2400, 2400):
                    raise ValueError(f"Unexpected dataset shape: {dataset.name}, {dataset.shape}.")
                if name.endswith("_Snow_Free"):
                    units = text_attribute(dataset.attrs, "units").replace(" ", "").replace("\n", "")
                    if units not in ("nWatts/(cm^2sr)", "nW/(cm^2sr)"):
                        raise ValueError(f"Unknown radiance units for {dataset.name}: {units}.")
                value = decode(dataset, src)
                if name not in arrays:
                    arrays[name] = np.full(bounds.shape, np.nan)
                arrays[name][dst] = value
                tile_info["datasets"][name] = {"dtype": str(dataset.dtype), "shape": list(dataset.shape),
                                                  "attributes": {k: json_value(val) for k,val in dataset.attrs.items()}}
            # Independent check of the file's coordinate arrays, where present.
            for coordinate, expected in (("lat", bounds.latitude[dst[0]]), ("lon", bounds.longitude[dst[1]])):
                if coordinate in fields:
                    coordinate_data = fields[coordinate]
                    if coordinate_data.ndim == 1:
                        part = np.asarray(coordinate_data[src[0] if coordinate == "lat" else src[1]])
                        target = expected
                    elif coordinate_data.ndim == 2:
                        part = np.asarray(coordinate_data[src])
                        target = expected[:, None] if coordinate == "lat" else expected[None, :]
                    else:
                        raise ValueError(f"Unexpected coordinate dimensions in {path.name}.")
                    # Actual VNP46A4.002 files label the upper-left pixel edges
                    # in lat/lon. Keep cell centres for finite-area integration,
                    # derived from checked tile bounds and 2400 grid dimensions.
                    edge_shift = 1/480 if coordinate == "lat" else -1/480
                    if np.allclose(part,target,rtol=0,atol=1e-5):
                        convention = "cell_centres"
                    elif np.allclose(part,target+edge_shift,rtol=0,atol=1e-5):
                        convention = "upper_left_edges"
                    else:
                        raise ValueError(f"Coordinate order/centres/edges disagree in {path.name}.")
                    tile_info["coordinate_conventions"][coordinate] = convention
            attributes[tile] = tile_info
        coverage[dst] += 1
    if not np.all(coverage == 1):
        raise ValueError("Incomplete or overlapping geographic coverage.")
    arrays["latitude"] = bounds.latitude
    arrays["longitude"] = bounds.longitude
    return arrays, attributes


def numeric_summary(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    return {"count": int(values.size), "min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.percentile(values,95)), "p99": float(np.percentile(values,99)),
            "max": float(values.max()), "mean": float(values.mean())}


def audit(arrays: dict, bounds: Bounds) -> dict:
    n = int(np.prod(bounds.shape))
    result = {"bounds": asdict(bounds), "crs": "EPSG:4326", "shape": list(bounds.shape),
              "total_pixels": n, "step_arcseconds": 15, "units": RADIANCE_UNIT,
              "radiance_to_W_m2_sr": 1e-5, "quality_policy": "Quality == 0, Num > 3, finite radiance >= 0",
              "land_water_counts": {}, "composites": {}}
    land = arrays["Land_Water_Mask"]
    for value in np.unique(land[np.isfinite(land)]):
        result["land_water_counts"][str(int(value))] = int(np.count_nonzero(land == value))
    result["land_water_counts"]["fill_or_nan"] = int((~np.isfinite(land)).sum())
    for prefix in COMPOSITES:
        key = prefix + "_Composite_Snow_Free"
        rad, qa, num, std = (arrays[key + suffix] for suffix in ("", "_Quality", "_Num", "_Std"))
        mask = good_mask(rad, qa, num)
        counts = {str(int(v)): int(np.count_nonzero(qa == v)) for v in np.unique(qa[np.isfinite(qa)])}
        counts["fill_or_nan"] = int(np.count_nonzero(~np.isfinite(qa)))
        valid_std = mask & np.isfinite(std) & (std >= 0)
        result["composites"][prefix] = {
            "denominator": n, "quality_counts": counts, "accepted_pixels": int(mask.sum()),
            "accepted_fraction": float(mask.mean()), "missing_radiance": int((~np.isfinite(rad)).sum()),
            "negative_nonfill_radiance": int((np.isfinite(rad) & (rad < 0)).sum()),
            "inconsistent_quality_vs_num": int(((qa == 0) & ~(num > 3)).sum()),
            "zero_pixels_accepted": int((mask & (rad == 0)).sum()),
            "radiance": numeric_summary(rad[mask]), "observations": numeric_summary(num[mask]),
            "std": numeric_summary(std[valid_std]),
            "missing_or_negative_std_accepted": int((mask & ~valid_std).sum()),
            "land_radiance": numeric_summary(rad[mask & np.isin(land,(0,1))]),
            "water_radiance": numeric_summary(rad[mask & np.isin(land,(2,3))]),
            "coastal_radiance": numeric_summary(rad[mask & (land == 5)]),
        }
    a, b = (p + "_Composite_Snow_Free" for p in ("NearNadir", "OffNadir"))
    common = good_mask(arrays[a], arrays[a+"_Quality"], arrays[a+"_Num"]) & good_mask(arrays[b], arrays[b+"_Quality"], arrays[b+"_Num"])
    positive = common & (arrays[a] > 0) & (arrays[b] > 0)
    result["angle_comparison"] = {"common_good_pixels": int(common.sum()), "positive_pair_pixels": int(positive.sum()),
                                  "off_over_near": numeric_summary(arrays[b][positive] / arrays[a][positive]),
                                  "interpretation": "Composite angular differences are not an identified emission function."}
    return result


def save_region(path: Path, arrays: dict, bounds: Bounds, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata_json=json.dumps({"bounds": asdict(bounds), **metadata}))


def load_region(path: Path) -> tuple[dict, Bounds, dict]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        arrays = {key: archive[key] for key in archive.files if key != "metadata_json"}
    return arrays, Bounds(**metadata["bounds"]), metadata
