"""Audit public native-band measurements without inventing clear-sky quality flags."""

import csv
from datetime import datetime, timezone
from pathlib import Path
import re
import zipfile

import numpy as np

from .data import numeric_summary
from .download import sha256
from .photometry import PhotometricBand, SurfaceBrightness

ASTMON_COLUMNS = {"Datetime_UTC", "Filter", "Magnitude", "Error"}
ASTMON_BANDS = ("Johnson-B", "Johnson-V", "Johnson-R")


def audit_astmon(path: Path) -> dict:
    values = {name: [] for name in ASTMON_BANDS}
    errors = {name: [] for name in ASTMON_BANDS}
    counts = {name: 0 for name in ASTMON_BANDS}
    finite_count = {name: 0 for name in ASTMON_BANDS}
    time_min = time_max = None
    spatial_header_warning = False
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        columns = reader.fieldnames or []
        if not ASTMON_COLUMNS.issubset(columns):
            raise ValueError("AstMon CSV lacks required datetime/filter/magnitude/error columns.")
        for number, row in enumerate(reader, 2):
            try:
                if None in row or any(row.get(k) is None for k in ASTMON_COLUMNS):
                    raise ValueError("wrong number of columns")
                band = row["Filter"]
                if band not in values:
                    raise ValueError("unknown native filter")
                stamp = datetime.strptime(row["Datetime_UTC"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                mag, error = float(row["Magnitude"]), float(row["Error"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid AstMon observation at CSV line {number}: {exc}") from exc
            time_min = stamp if time_min is None else min(time_min, stamp)
            time_max = stamp if time_max is None else max(time_max, stamp)
            counts[band] += 1
            spatial_header_warning |= row.get("Pixel_x", "").startswith("Pos")
            if np.isfinite(mag) and np.isfinite(error):
                finite_count[band] += 1
                if error > 0:
                    values[band].append(mag)
                    errors[band].append(error)
    if not sum(counts.values()):
        raise ValueError("AstMon CSV contains no observations.")
    bands = {}
    for name in ASTMON_BANDS:
        mags = np.asarray(values[name])
        band = PhotometricBand(name, "not-explicit-in-CSV", "doi:10.5281/zenodo.4633001")
        brightness = SurfaceBrightness(mags, band) if len(mags) else None
        bands[name] = {
            "rows": counts[name], "finite_magnitude_and_error": finite_count[name],
            "finite_with_positive_reported_error": len(mags),
            "native_mag_arcsec2": numeric_summary(mags),
            "reported_error_mag": numeric_summary(np.asarray(errors[name])),
            "brightest_over_faintest_flux_ratio": float(
                SurfaceBrightness(mags.min(), band).ratio_to(SurfaceBrightness(mags.max(), band)))
                if brightness is not None else None,
        }
    return {
        "dataset": "AstMon_evol.csv, Madrid zenith time series", "filename": path.name,
        "sha256": sha256(path), "columns": columns, "rows": sum(counts.values()),
        "utc_start": time_min.isoformat(), "utc_end": time_max.isoformat(), "bands": bands,
        "clear_sky_flags_available": False, "moon_flags_available": False,
        "rows_confirmed_clear_and_moonless_by_this_import": 0,
        "all_sky_maps_in_this_file": False, "absolute_radiance_conversion_applied": False,
        "spatial_header_warning": spatial_header_warning,
        "limitations": [
            "Finite values and positive reported errors are numerical checks, not weather QA.",
            "Native B, V, R samples are sequential; colour indices require time matching.",
            "No natural/artificial/moonlight decomposition, sensor QE or full camera calibration.",
            "Pixel_x contains Pos labels in this release; do not use these columns as astrometric coordinates.",
        ],
    }


def audit_galicia(path: Path) -> dict:
    sites = []
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > 512*1024*1024:
            raise ValueError("Reference archive exceeds the supported uncompressed size.")
        offsets_text = archive.read("Offsets.txt").decode("utf-8-sig")
        pairs = re.findall(r"strcmp\(BaseNome,'([^']+)'\)\s+Offset=([+-]?\d+(?:\.\d+)?)", offsets_text)
        offsets = {name: float(value) for name, value in pairs}
        if not offsets or len(offsets) != len(pairs):
            raise ValueError("Missing or ambiguous reference-detector offsets.")
        names = sorted(n for n in archive.namelist() if re.fullmatch(r"\d{2}_[^/]+\.txt", n))
        for name in names:
            if Path(name).stem not in offsets:
                raise ValueError(f"No reference offset for {name}.")
            with archive.open(name) as stream:
                data = np.loadtxt(stream, ndmin=2)
            if data.shape[1] != 7 or not np.all(np.isfinite(data[:, :6])):
                raise ValueError(f"Invalid seven-column time series in {name}.")
            dates = data[:, :6]
            if (np.any(dates != np.floor(dates)) or np.any((dates[:, 0] < 1) | (dates[:, 0] > 31))
                    or np.any((dates[:, 1] < 1) | (dates[:, 1] > 12))
                    or np.any((dates[:, 3:] < 0) | (dates[:, 3:] >= [24, 60, 60]))):
                raise ValueError(f"Invalid UTC timestamp fields in {name}.")
            raw = data[:, 6]
            valid = np.isfinite(raw) & (raw > 0)
            offset = offsets[Path(name).stem]
            corrected = raw[valid]-offset
            def timestamp(row):
                day, month, year, hour, minute, second = map(int, row[:6])
                return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).isoformat()
            sites.append({"site": Path(name).stem, "rows": len(raw),
                          "positive_finite_native_readings": int(valid.sum()),
                          "nonpositive_or_nonfinite_readings": int((~valid).sum()),
                          "offset_subtracted_mag": offset,
                          "first_row_utc": timestamp(data[0]), "last_row_utc": timestamp(data[-1]),
                          "corrected_native_mag_arcsec2": numeric_summary(corrected)})
    if not sites:
        raise ValueError("Reference archive has no site time series.")
    return {"dataset": "GalicianNightSkyBrightnessData_2015", "filename": path.name,
            "sha256": sha256(path), "sites": sites, "site_count": len(sites),
            "rows": sum(s["rows"] for s in sites),
            "positive_finite_native_readings": sum(s["positive_finite_native_readings"] for s in sites),
            "band": "SQM network common reference; not Johnson V or CIE scotopic",
            "offset_rule": "mag_corrected = mag_raw - offset; never apply to placeholder values",
            "clear_sky_flags_available": False, "moon_flags_available": False,
            "limitations": ["Nonpositive readings excluded by project QC; never interpreted as zero light.",
                            "Daylight, cloud, moon and instrument-state selection still required.",
                            "Relative network offsets are not a conversion to spectral radiance."]}
