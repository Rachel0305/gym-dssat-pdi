"""Read-only, low-memory GPCC/CPC precipitation sanity audit for Task 023."""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "hl_fq_global_precip_023"
DOC = ROOT / "docs" / "hl_fq_global_precip_sanity_audit_023.md"
IDENTITY_DOC = ROOT / "docs" / "hl_fq_global_precip_product_identity_023.md"
GPCC_DIR = ROOT / "data" / "external" / "gpcc_daily" / "raw"
CPC_DIR = ROOT / "data" / "external" / "cpc_global_precip" / "raw"
YEARS = (2011, 2012, 2013)
PRODUCTS = ("GPCC", "CPC")
STRONG_RAIN_MM = 50.0  # transparent regional-event diagnostic, not a validation threshold

GPCC_FILES = {year: GPCC_DIR / f"full_data_daily_v2022_10_{year}.nc.gz" for year in YEARS}
CPC_FILES = {year: CPC_DIR / f"precip.{year}.nc" for year in YEARS}
D222_CSV = ROOT / "results" / "five_site_d222_version_audit_022_rerun_20261007" / "current_d222_daily_normalized.csv"
D222_IDENTITY = ROOT / "results" / "five_site_d222_version_audit_022_rerun_20261007" / "new_d222_snapshot_identity.json"
HLA_UNRESOLVED_CSV = ROOT / "results" / "hla_random_weather_8seed" / "014_isd50756_recovery" / "unresolved_dates_after_014.csv"
FQA_AUDIT_JSON = ROOT / "results" / "hl_fq_random_weather_8seed" / "source_audit" / "weather_source_audit_final.json"
HLA_ALT_IDENTITY = ROOT / "results" / "hl_fq_isdlite_021" / "hla_station_identity.json"
COORD_WTH = {"HLA": ROOT / "my_data" / "CNHL0701.WTH", "FQA": ROOT / "my_data" / "CNFQ0701.WTH"}
EXPECTED_WTH_STATION = {"HLA": "CNHL", "FQA": "CNFQ"}
EVENT_DATES = {
    "HLA": ["2013-07-30", "2013-08-09", "2013-08-12", "2013-07-21", "2013-08-21"],
    "FQA": ["2012-09-02", "2013-05-26"],
}

OUTPUT_NAMES = [
    "input_provenance.json", "site_coordinates.json", "product_identity.md",
    "hla_gpcc_daily_2011_2013.csv", "hla_cpc_daily_2011_2013.csv",
    "fqa_gpcc_daily_2011_2013.csv", "fqa_cpc_daily_2011_2013.csv",
    "hla_2013_sanity.csv", "fqa_2011_2013_sanity.csv",
    "gpcc_vs_cpc_comparison.csv", "final_gate.json",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def rounded(value: float | int | None, digits: int = 4) -> float | str:
    if value is None:
        return ""
    value = float(value)
    return round(value, digits) if math.isfinite(value) else ""


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def read_wth_coordinate(path: Path, site: str) -> dict[str, Any]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.strip().upper().startswith("@ INSI")), None)
    if header_index is None or header_index + 1 >= len(lines):
        raise ValueError(f"WTH coordinate header not found: {path}")
    header = lines[header_index].replace("@", " ").split()
    values = lines[header_index + 1].split()
    lookup = {name.upper(): index for index, name in enumerate(header)}
    if values[0].upper() != EXPECTED_WTH_STATION[site]:
        raise ValueError(f"Unexpected station code in {path}: {values[0]}")
    lat = float(values[lookup["LAT"]])
    lon = float(values[lookup["LONG"]])
    elev = float(values[lookup["ELEV"]]) if "ELEV" in lookup else None
    return {
        "site": site,
        "station_id_in_wth": values[0],
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elev,
        "coordinate_source_path": rel(path),
        "coordinate_source_line": header_index + 2,
        "source_type": "project station WTH header",
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians((lon2 - lon1 + 180.0) % 360.0 - 180.0)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(a))


def locate_nearest(latitudes: np.ndarray, longitudes: np.ndarray, site: dict[str, Any]) -> dict[str, Any]:
    lat_index = int(np.argmin(np.abs(latitudes.astype(float) - site["latitude"])))
    lon = site["longitude"]
    lon_array = longitudes.astype(float)
    if float(np.nanmin(lon_array)) >= 0 and lon > 180:
        target_lon = lon % 360.0
    elif float(np.nanmin(lon_array)) >= 0:
        target_lon = lon % 360.0
    else:
        target_lon = (lon + 180.0) % 360.0 - 180.0
    lon_delta = np.abs((lon_array - target_lon + 180.0) % 360.0 - 180.0)
    lon_index = int(np.argmin(lon_delta))
    center_lat = float(latitudes[lat_index])
    center_lon = float(longitudes[lon_index])
    return {
        "lat_index": lat_index,
        "lon_index": lon_index,
        "grid_latitude": center_lat,
        "grid_longitude": center_lon,
        "nearest_grid_distance_km": haversine_km(site["latitude"], site["longitude"], center_lat, center_lon),
        "lat_indices": list(range(max(0, lat_index - 1), min(len(latitudes), lat_index + 2))),
        "lon_indices": [(lon_index + offset) % len(longitudes) for offset in (-1, 0, 1)],
    }


# Minimal CDF-1 parser. It reads the header and then streams complete records;
# no decompressed annual file or multi-variable global array is written to disk/RAM.
CDF_TYPE = {
    1: ("byte", 1, ">i1"), 2: ("char", 1, "S1"), 3: ("short", 2, ">i2"),
    4: ("int", 4, ">i4"), 5: ("float", 4, ">f4"), 6: ("double", 8, ">f8"),
}


def read_exact(stream: Any, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise EOFError(f"Expected {count} bytes, got {len(data)}")
    return data


def u32(stream: Any) -> int:
    return struct.unpack(">I", read_exact(stream, 4))[0]


def read_cdf_name(stream: Any) -> str:
    length = u32(stream)
    raw = read_exact(stream, length)
    padding = (-length) % 4
    if padding:
        read_exact(stream, padding)
    return raw.decode("utf-8", errors="replace")


def read_cdf_list_header(stream: Any, expected_tag: int) -> int:
    tag = u32(stream)
    count = u32(stream)
    if tag == 0 and count == 0:
        return 0
    if tag != expected_tag:
        raise ValueError(f"Unexpected CDF list tag {tag}; expected {expected_tag}")
    return count


def decode_cdf_attribute(type_id: int, count: int, raw: bytes) -> Any:
    name, itemsize, dtype = CDF_TYPE[type_id]
    if name == "char":
        return raw.decode("utf-8", errors="replace").rstrip("\x00")
    values = np.frombuffer(raw, dtype=np.dtype(dtype), count=count)
    return json_value(values[0] if count == 1 else values)


def read_cdf_attributes(stream: Any) -> dict[str, Any]:
    count = read_cdf_list_header(stream, 12)  # NC_ATTRIBUTE
    result = {}
    for _ in range(count):
        name = read_cdf_name(stream)
        type_id = u32(stream)
        nvalues = u32(stream)
        if type_id not in CDF_TYPE:
            raise ValueError(f"Unknown CDF attribute type {type_id}")
        itemsize = CDF_TYPE[type_id][1]
        byte_count = nvalues * itemsize
        raw = read_exact(stream, byte_count)
        padding = (-byte_count) % 4
        if padding:
            read_exact(stream, padding)
        result[name] = decode_cdf_attribute(type_id, nvalues, raw)
    return result


def parse_gpcc_cdf_header(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        magic = read_exact(stream, 4)
        if magic not in (b"CDF\x01", b"CDF\x02"):
            raise ValueError(f"Unsupported NetCDF signature in {path}: {magic!r}")
        version = magic[-1]
        nrecords = u32(stream)
        dimension_count = read_cdf_list_header(stream, 10)  # NC_DIMENSION
        dimensions: list[dict[str, Any]] = []
        for _ in range(dimension_count):
            name = read_cdf_name(stream)
            length = u32(stream)
            dimensions.append({"name": name, "length_in_header": length})
        global_attrs = read_cdf_attributes(stream)
        variable_count = read_cdf_list_header(stream, 11)  # NC_VARIABLE
        variables: list[dict[str, Any]] = []
        for _ in range(variable_count):
            name = read_cdf_name(stream)
            dimension_ids = [u32(stream) for _ in range(u32(stream))]
            attrs = read_cdf_attributes(stream)
            type_id = u32(stream)
            size = CDF_TYPE[type_id][1]
            vsize = u32(stream)
            begin = struct.unpack(">Q" if version == 2 else ">I", read_exact(stream, 8 if version == 2 else 4))[0]
            dim_names = [dimensions[index]["name"] for index in dimension_ids]
            shape = []
            for dim_id in dimension_ids:
                dim = dimensions[dim_id]
                shape.append(nrecords if dim["length_in_header"] == 0 else dim["length_in_header"])
            variables.append({
                "name": name, "dimension_names": dim_names, "shape": shape,
                "type_id": type_id, "type_name": CDF_TYPE[type_id][0],
                "item_size": size, "vsize": vsize, "begin": begin, "attributes": attrs,
            })
        unlim = next((d["name"] for d in dimensions if d["length_in_header"] == 0), None)
    variable_map = {item["name"]: item for item in variables}
    record_vars = [item for item in variables if item["dimension_names"] and item["dimension_names"][0] == unlim]
    if not record_vars:
        raise ValueError(f"No record variables in {path}")
    record_start = min(item["begin"] for item in record_vars)
    record_size = sum(item["vsize"] for item in record_vars)
    for item in record_vars:
        item["record_offset"] = item["begin"] - record_start
    if "precip" not in variable_map or "time" not in variable_map:
        raise ValueError(f"Missing precip/time variable in {path}")
    with path.open("rb") as raw:
        raw.seek(-4, 2)
        gzip_isize_mod_4gb = struct.unpack("<I", raw.read(4))[0]
    return {
        "format": "NetCDF classic CDF-1" if version == 1 else "NetCDF 64-bit offset CDF-2",
        "version": version,
        "num_records": nrecords,
        "dimensions": dimensions,
        "global_attributes": global_attrs,
        "variables": variables,
        "variable_map": variable_map,
        "record_variables": record_vars,
        "record_start": record_start,
        "record_size": record_size,
        "uncompressed_size_mod_4gb": gzip_isize_mod_4gb,
    }


def read_fixed_cdf_variable(path: Path, variable: dict[str, Any]) -> np.ndarray:
    if variable["dimension_names"] and variable["dimension_names"][0] == "time":
        raise ValueError("Expected fixed-size coordinate variable")
    type_id = variable["type_id"]
    count = math.prod(variable["shape"])
    byte_count = count * variable["item_size"]
    with gzip.open(path, "rb") as stream:
        stream.seek(variable["begin"])
        raw = read_exact(stream, byte_count)
    array = np.frombuffer(raw, dtype=np.dtype(CDF_TYPE[type_id][2]), count=count)
    return array.reshape(variable["shape"]).astype(float, copy=True)


def valid_grid_value(value: float, missing: Any) -> float | None:
    if not math.isfinite(float(value)):
        return None
    if missing is not None and float(value) == float(missing):
        return None
    if float(value) < -1000:
        return None
    return float(value)


def summarize_patch(field: np.ndarray, grid: dict[str, Any], missing: Any) -> dict[str, Any]:
    indices = np.ix_(grid["lat_indices"], grid["lon_indices"])
    patch = np.asarray(field[indices], dtype=float)
    values = [valid_grid_value(value, missing) for value in patch.ravel()]
    valid = [value for value in values if value is not None]
    center = valid_grid_value(float(field[grid["lat_index"], grid["lon_index"]]), missing)
    return {
        "nearest_grid_precip_mm": rounded(center),
        "neighbor_mean_mm": rounded(float(np.mean(valid)) if valid else None),
        "neighbor_median_mm": rounded(float(np.median(valid)) if valid else None),
        "neighbor_min_mm": rounded(min(valid) if valid else None),
        "neighbor_max_mm": rounded(max(valid) if valid else None),
        "neighbor_valid_cells": len(valid),
        "nearest_grid_latitude": grid["grid_latitude"],
        "nearest_grid_longitude": grid["grid_longitude"],
        "nearest_grid_distance_km": rounded(grid["nearest_grid_distance_km"], 3),
    }


def gpcc_daily_extract(path: Path, year: int, site_coordinates: dict[str, Any], gpcc_meta: dict[str, Any]) -> dict[str, dict[dt.date, dict[str, Any]]]:
    header = parse_gpcc_cdf_header(path)
    variables = header["variable_map"]
    latitudes = read_fixed_cdf_variable(path, variables["lat"]).ravel()
    longitudes = read_fixed_cdf_variable(path, variables["lon"]).ravel()
    grids = {site: locate_nearest(latitudes, longitudes, coord) for site, coord in site_coordinates.items()}
    time_var = variables["time"]
    time_units = decode_text(time_var["attributes"].get("units", ""))
    match = re.search(r"days since (\d{4})-(\d{1,2})-(\d{1,2})", time_units)
    if not match:
        raise ValueError(f"Unsupported GPCC time units {time_units!r}")
    time_origin = dt.date(*(int(value) for value in match.groups()))
    rec_offsets = {item["name"]: item["record_offset"] for item in header["record_variables"]}
    record_size = header["record_size"]
    needed = ("time", "precip", "interpolation_error", "numgauge")
    for name in needed:
        if name not in rec_offsets:
            raise ValueError(f"GPCC variable {name} is not a streamed record variable")
    precip_var = variables["precip"]
    error_var = variables["interpolation_error"]
    gauge_var = variables["numgauge"]
    lat_count, lon_count = precip_var["shape"][1:]
    cell_count = lat_count * lon_count
    missing_precip = precip_var["attributes"].get("missing_value", precip_var["attributes"].get("_FillValue", -9999.0))
    missing_error = error_var["attributes"].get("missing_value", error_var["attributes"].get("_FillValue", -9999.0))
    missing_gauge = gauge_var["attributes"].get("missing_value", gauge_var["attributes"].get("_FillValue", -9999.0))
    result = {site: {} for site in site_coordinates}
    with gzip.open(path, "rb") as stream:
        stream.seek(header["record_start"])
        for index in range(header["num_records"]):
            record = read_exact(stream, record_size)
            day_number = struct.unpack_from(">d", record, rec_offsets["time"])[0]
            date = time_origin + dt.timedelta(days=day_number)
            expected = dt.date(year, 1, 1) + dt.timedelta(days=index)
            if date != expected:
                raise ValueError(f"GPCC time mismatch in {path.name}: index {index} is {date}, expected {expected}")
            precip = np.frombuffer(record, dtype=">f4", count=cell_count, offset=rec_offsets["precip"]).reshape(lat_count, lon_count)
            interp = np.frombuffer(record, dtype=">f4", count=cell_count, offset=rec_offsets["interpolation_error"]).reshape(lat_count, lon_count)
            gauges = np.frombuffer(record, dtype=">f4", count=cell_count, offset=rec_offsets["numgauge"]).reshape(lat_count, lon_count)
            for site, grid in grids.items():
                row = summarize_patch(precip, grid, missing_precip)
                err = valid_grid_value(float(interp[grid["lat_index"], grid["lon_index"]]), missing_error)
                ngauge = valid_grid_value(float(gauges[grid["lat_index"], grid["lon_index"]]), missing_gauge)
                row["interpolation_error_nearest_mm"] = rounded(err)
                row["numgauge_nearest"] = rounded(ngauge, 1)
                row["quality_auxiliary_status"] = "AVAILABLE_IN_GPCC_FILE"
                row["product"] = "GPCC"
                row["site"] = site
                row["date"] = date
                result[site][date] = row
    return result


def h5_attrs(dataset: Any) -> dict[str, Any]:
    excluded = {"DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME", "_Netcdf4Dimid"}
    return {str(key): json_value(value) for key, value in dataset.attrs.items() if key not in excluded}


def parse_time_origin(units: str) -> tuple[dt.datetime, str]:
    match = re.search(r"hours since (\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{1,2}):(\d{1,2}))?", units)
    if not match:
        raise ValueError(f"Unsupported CPC time units {units!r}")
    parts = [int(value or 0) for value in match.groups()]
    return dt.datetime(*parts), "standard (CF default; no explicit calendar attribute)"


def cpc_daily_extract(path: Path, year: int, site_coordinates: dict[str, Any]) -> tuple[dict[str, dict[dt.date, dict[str, Any]]], dict[str, Any]]:
    result = {site: {} for site in site_coordinates}
    with h5py.File(path, "r") as nc:
        latitudes = np.asarray(nc["lat"][:], dtype=float)
        longitudes = np.asarray(nc["lon"][:], dtype=float)
        dataset = nc["precip"]
        time_var = nc["time"]
        time_units = decode_text(time_var.attrs["units"])
        origin, calendar_description = parse_time_origin(time_units)
        time_values = np.asarray(time_var[:], dtype=float)
        grids = {site: locate_nearest(latitudes, longitudes, coord) for site, coord in site_coordinates.items()}
        missing = float(np.asarray(dataset.attrs.get("missing_value", -9.96921e36)).ravel()[0])
        for index, hour_value in enumerate(time_values):
            timestamp = origin + dt.timedelta(hours=float(hour_value))
            date = timestamp.date()
            expected = dt.date(year, 1, 1) + dt.timedelta(days=index)
            if date != expected or timestamp.hour or timestamp.minute or timestamp.second:
                raise ValueError(f"CPC time mismatch in {path.name}: index {index} is {timestamp}, expected {expected} 00:00")
            # CPC is chunked as one compressed global daily field. Read one day
            # into a small (~1 MB) array once, then sample both sites.
            daily_field = np.asarray(dataset[index, :, :], dtype=float)
            for site, grid in grids.items():
                patch = daily_field[np.ix_(grid["lat_indices"], grid["lon_indices"])]
                valid = [valid_grid_value(float(value), missing) for value in patch.ravel()]
                valid = [value for value in valid if value is not None]
                center = valid_grid_value(float(daily_field[grid["lat_index"], grid["lon_index"]]), missing)
                result[site][date] = {
                    "nearest_grid_precip_mm": rounded(center),
                    "neighbor_mean_mm": rounded(float(np.mean(valid)) if valid else None),
                    "neighbor_median_mm": rounded(float(np.median(valid)) if valid else None),
                    "neighbor_min_mm": rounded(min(valid) if valid else None),
                    "neighbor_max_mm": rounded(max(valid) if valid else None),
                    "neighbor_valid_cells": len(valid),
                    "nearest_grid_latitude": grid["grid_latitude"],
                    "nearest_grid_longitude": grid["grid_longitude"],
                    "nearest_grid_distance_km": rounded(grid["nearest_grid_distance_km"], 3),
                    "interpolation_error_nearest_mm": "",
                    "numgauge_nearest": "",
                    "quality_auxiliary_status": "NOT_PRESENT_IN_SELECTED_NETCDF_FILE",
                    "product": "CPC",
                    "site": site,
                    "date": date,
                }
            del daily_field
        metadata = {
            "format": "NetCDF-4/HDF5",
            "global_attributes": {str(k): json_value(v) for k, v in nc.attrs.items()},
            "dimensions": {name: int(nc[name].shape[0]) for name in ("time", "lat", "lon")},
            "variables": {
                name: {
                    "shape": list(nc[name].shape),
                    "dtype": str(nc[name].dtype),
                    "chunks": list(nc[name].chunks) if nc[name].chunks else None,
                    "compression": nc[name].compression,
                    "attributes": h5_attrs(nc[name]),
                }
                for name in ("lat", "lon", "time", "precip")
            },
            "time_units": time_units,
            "calendar": calendar_description,
            "date_label": "time coordinates are at 00:00 and marked coordinate_defines=start; NOAA documentation says the daily end-of-day time varies by country, not UTC-uniform.",
            "latitude_centers": [float(np.min(latitudes)), float(np.max(latitudes))],
            "longitude_centers": [float(np.min(longitudes)), float(np.max(longitudes))],
            "latitude_resolution_degrees": rounded(float(np.median(np.abs(np.diff(latitudes)))), 6),
            "longitude_resolution_degrees": rounded(float(np.median(np.abs(np.diff(longitudes)))), 6),
        }
    return result, metadata


def grid_product_metadata(path: Path, header: dict[str, Any], latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, Any]:
    variables = {}
    for item in header["variables"]:
        variables[item["name"]] = {
            "dimensions": item["dimension_names"], "shape": item["shape"],
            "type": item["type_name"], "attributes": item["attributes"],
        }
    return {
        "format": header["format"], "global_attributes": header["global_attributes"],
        "dimensions": {item["name"]: (header["num_records"] if item["length_in_header"] == 0 else item["length_in_header"]) for item in header["dimensions"]},
        "variables": variables,
        "latitude_centers": [float(np.min(latitudes)), float(np.max(latitudes))],
        "longitude_centers": [float(np.min(longitudes)), float(np.max(longitudes))],
        "latitude_resolution_degrees": rounded(float(np.median(np.abs(np.diff(latitudes)))), 6),
        "longitude_resolution_degrees": rounded(float(np.median(np.abs(np.diff(longitudes)))), 6),
        "time_units": header["variable_map"]["time"]["attributes"].get("units", ""),
        "calendar": header["variable_map"]["time"]["attributes"].get("calendar", "not stated"),
        "date_label": "NetCDF daily time coordinate is a calendar-date label at 00:00; no UTC/time-zone or per-cell observing-day boundary is encoded in this file.",
        "uncompressed_size_mod_4gb": header["uncompressed_size_mod_4gb"],
    }


def input_file_record(path: Path, role: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": rel(path), "path_absolute": str(path.resolve()), "filename": path.name,
        "size_bytes": stat.st_size,
        "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
        "sha256": sha256(path), "role": role,
    }


def site_product_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": record["date"].isoformat(), "site": record["site"], "product": record["product"],
        "nearest_grid_precip_mm": record["nearest_grid_precip_mm"],
        "neighbor_mean_mm": record["neighbor_mean_mm"],
        "neighbor_median_mm": record["neighbor_median_mm"],
        "neighbor_min_mm": record["neighbor_min_mm"], "neighbor_max_mm": record["neighbor_max_mm"],
        "neighbor_valid_cells": record["neighbor_valid_cells"],
        "nearest_grid_latitude": record["nearest_grid_latitude"],
        "nearest_grid_longitude": record["nearest_grid_longitude"],
        "nearest_grid_distance_km": record["nearest_grid_distance_km"],
        "interpolation_error_nearest_mm": record["interpolation_error_nearest_mm"],
        "numgauge_nearest": record["numgauge_nearest"],
        "quality_auxiliary_status": record["quality_auxiliary_status"],
    }


def parse_d222_inputs() -> tuple[dict[str, dict[dt.date, dict[str, Any]]], set[str], dict[str, Any]]:
    identity = json.loads(D222_IDENTITY.read_text(encoding="utf-8"))
    snapshot_id = identity.get("snapshot_id", "CERN_D222_CURRENT_SNAPSHOT_FROM_TASK_022")
    site_dates: dict[str, dict[dt.date, dict[str, Any]]] = {site: {} for site in ("HLA", "FQA")}
    with D222_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            site = row["project_site"]
            if site not in site_dates:
                continue
            date = dt.date.fromisoformat(row["date"])
            if date.year not in YEARS:
                continue
            value = float(row["rain_mm"]) if row["rain_mm"] else None
            site_dates[site][date] = {"rain_mm": value, "value_status": row["value_status"]}
    hla_unresolved_rows = read_csv(HLA_UNRESOLVED_CSV)
    hla_unresolved = {row["date"] for row in hla_unresolved_rows}
    fqa_audit = json.loads(FQA_AUDIT_JSON.read_text(encoding="utf-8"))
    fqa_unresolved = set(fqa_audit["sites"]["FQA"]["official_ghcn_rain_screen"]["unresolved_local_rain_days"])
    return site_dates, hla_unresolved, {
        "snapshot_id": snapshot_id,
        "hla_unresolved_dates": hla_unresolved,
        "fqa_unresolved_dates": fqa_unresolved,
        "hla_unresolved_source": rel(HLA_UNRESOLVED_CSV),
        "fqa_unresolved_source": rel(FQA_AUDIT_JSON),
    }


def numeric_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def pair_metrics(x_values: list[float], y_values: list[float]) -> dict[str, Any]:
    pairs = [(float(x), float(y)) for x, y in zip(x_values, y_values) if math.isfinite(x) and math.isfinite(y)]
    n = len(pairs)
    result: dict[str, Any] = {"overlap_days": n}
    if not pairs:
        return {**result, "wet_dry_agreement_rain_gt_0": "", "wet_dry_agreement_rain_ge_1mm": "", "mae_mm": "", "rmse_mm": "", "mean_bias_product_minus_reference_mm": "", "median_bias_product_minus_reference_mm": "", "pearson_r": "", "spearman_rho": ""}
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    delta = y - x
    wet0 = (x > 0) == (y > 0)
    wet1 = (x >= 1) == (y >= 1)
    pearson = float(np.corrcoef(x, y)[0, 1]) if n >= 2 and np.std(x) > 0 and np.std(y) > 0 else None
    spearman = float(stats.spearmanr(x, y).statistic) if n >= 2 and np.std(x) > 0 and np.std(y) > 0 else None
    result.update({
        "wet_dry_agreement_rain_gt_0": rounded(float(np.mean(wet0)), 6),
        "wet_dry_agreement_rain_ge_1mm": rounded(float(np.mean(wet1)), 6),
        "mae_mm": rounded(float(np.mean(np.abs(delta))), 4),
        "rmse_mm": rounded(float(np.sqrt(np.mean(delta ** 2))), 4),
        "mean_bias_product_minus_reference_mm": rounded(float(np.mean(delta)), 4),
        "median_bias_product_minus_reference_mm": rounded(float(np.median(delta)), 4),
        "pearson_r": rounded(pearson, 6), "spearman_rho": rounded(spearman, 6),
    })
    return result


def period_dates(year: int, month: int | None = None) -> list[dt.date]:
    if month is None:
        start = dt.date(year, 1, 1)
        count = (dt.date(year + 1, 1, 1) - start).days
    else:
        start = dt.date(year, month, 1)
        end = dt.date(year + 1, 1, 1) if month == 12 else dt.date(year, month + 1, 1)
        count = (end - start).days
    return [start + dt.timedelta(days=offset) for offset in range(count)]


def site_numeric_summary(site: str, dates: list[dt.date], d222: dict[str, dict[dt.date, dict[str, Any]]]) -> dict[str, Any]:
    records = d222[site]
    numeric = [records[date]["rain_mm"] for date in dates if date in records and records[date]["rain_mm"] is not None]
    blank_rows = sum(date in records and records[date]["rain_mm"] is None for date in dates)
    missing_dates = sum(date not in records for date in dates)
    expected = len(dates)
    return {
        "expected_days": expected,
        "d222_record_days": expected - missing_dates,
        "d222_numeric_days": len(numeric),
        "d222_blank_row_days": blank_rows,
        "d222_absent_date_days": missing_dates,
        "d222_missing_days": expected - len(numeric),
        "d222_numeric_subtotal_mm": rounded(sum(numeric), 3),
        "d222_annual_total_mm": rounded(sum(numeric), 3) if len(numeric) == expected else "",
        "d222_completeness_fraction": rounded(len(numeric) / expected if expected else None, 6),
    }


def product_period_summary(site: str, product: str, dates: list[dt.date], product_days: dict[str, Any], d222_site: dict[dt.date, dict[str, Any]]) -> dict[str, Any]:
    items = [product_days[product][site].get(date) for date in dates]
    nearest = [numeric_or_none(item["nearest_grid_precip_mm"]) if item else None for item in items]
    nvalid = sum(value is not None for value in nearest)
    valid_sum = sum(value for value in nearest if value is not None)
    full_total = rounded(valid_sum, 3) if nvalid == len(dates) else ""
    common_pairs = [(d222_site[date]["rain_mm"], nearest[i]) for i, date in enumerate(dates) if date in d222_site and d222_site[date]["rain_mm"] is not None and nearest[i] is not None]
    d222_common = sum(pair[0] for pair in common_pairs)
    product_common = sum(pair[1] for pair in common_pairs)
    bias_ratio = (d222_common - product_common) / product_common if product_common != 0 else None
    return {
        "product_valid_days": nvalid,
        "product_valid_day_subtotal_mm": rounded(valid_sum, 3),
        "product_full_period_total_mm": full_total,
        "d222_product_common_days": len(common_pairs),
        "d222_common_day_subtotal_mm": rounded(d222_common, 3),
        "product_common_day_subtotal_mm": rounded(product_common, 3),
        "common_day_bias_ratio_d222_minus_product": rounded(bias_ratio, 6),
    }


def product_event_summary(site: str, product: str, event: dt.date, product_days: dict[str, Any]) -> dict[str, Any]:
    daily = product_days[product][site]
    current = daily.get(event, {})
    def window(days: int) -> list[dict[str, Any]]:
        return [daily.get(event + dt.timedelta(offset), {}) for offset in range(-days, days + 1)]
    def max_field(rows: list[dict[str, Any]], key: str) -> Any:
        values = [numeric_or_none(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        return rounded(max(values), 3) if values else ""
    w1, w2 = window(1), window(2)
    nday = numeric_or_none(current.get("nearest_grid_precip_mm"))
    neighbor_max = numeric_or_none(current.get("neighbor_max_mm"))
    return {
        f"{product.lower()}_nearest_mm": current.get("nearest_grid_precip_mm", ""),
        f"{product.lower()}_3x3_mean_mm": current.get("neighbor_mean_mm", ""),
        f"{product.lower()}_3x3_min_mm": current.get("neighbor_min_mm", ""),
        f"{product.lower()}_3x3_max_mm": current.get("neighbor_max_mm", ""),
        f"{product.lower()}_max_nearest_pm1_mm": max_field(w1, "nearest_grid_precip_mm"),
        f"{product.lower()}_max_nearest_pm2_mm": max_field(w2, "nearest_grid_precip_mm"),
        f"{product.lower()}_max_3x3_pm1_mm": max_field(w1, "neighbor_max_mm"),
        f"{product.lower()}_max_3x3_pm2_mm": max_field(w2, "neighbor_max_mm"),
        f"{product.lower()}_strong_region_same_day_ge50": neighbor_max is not None and neighbor_max >= STRONG_RAIN_MM,
        f"{product.lower()}_strong_region_pm1_ge50": numeric_or_none(max_field(w1, "neighbor_max_mm")) is not None and float(max_field(w1, "neighbor_max_mm")) >= STRONG_RAIN_MM,
        f"{product.lower()}_strong_region_pm2_ge50": numeric_or_none(max_field(w2, "neighbor_max_mm")) is not None and float(max_field(w2, "neighbor_max_mm")) >= STRONG_RAIN_MM,
        f"{product.lower()}_nearest_grid_daily_value_valid": nday is not None,
    }


def append_shift_diagnostics(rows: list[dict[str, Any]], site: str, year: int, d222: dict[str, dict[dt.date, dict[str, Any]]], product_days: dict[str, Any]) -> None:
    for product in PRODUCTS:
        for shift in (-1, 0, 1):
            dvals, pvals = [], []
            for date in period_dates(year):
                drecord = d222[site].get(date)
                pdate = date + dt.timedelta(days=shift)
                pvalue = numeric_or_none(product_days[product][site].get(pdate, {}).get("nearest_grid_precip_mm"))
                if drecord and drecord["rain_mm"] is not None and pvalue is not None:
                    dvals.append(drecord["rain_mm"])
                    pvals.append(pvalue)
            metrics = pair_metrics(dvals, pvals)
            rows.append({
                "row_type": "D222_TIME_SHIFT_DIAGNOSTIC",
                "site": site, "year": year, "product": product,
                "shift_days_product_date_relative_to_d222": shift,
                "shift_definition": "compare D222(date) with gridded product(date + shift); diagnostic only",
                **metrics,
            })


def create_sanity_outputs(d222: dict[str, dict[dt.date, dict[str, Any]]], product_days: dict[str, Any], unresolved: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    hla_rows: list[dict[str, Any]] = []
    fqa_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []

    # HLA 2013 daily rows and unresolved reference-only signals.
    for date in period_dates(2013):
        d = d222["HLA"].get(date, {"rain_mm": None, "value_status": "DATE_ABSENT"})
        is_unresolved = date.isoformat() in unresolved["hla_unresolved_dates"]
        row = {
            "row_type": "DAILY_2013",
            "date": date.isoformat(), "site": "HLA", "d222_rain_mm": rounded(d["rain_mm"], 3),
            "d222_value_status": d["value_status"], "previous_014_unresolved": is_unresolved,
            "global_product_role": "REFERENCE_ONLY" if is_unresolved else "COMPARISON_ON_EXPLICIT_D222_VALUE",
        }
        for product in PRODUCTS:
            item = site_product_row(product_days[product]["HLA"][date])
            for key, value in item.items():
                if key not in ("date", "site", "product"):
                    row[f"{product.lower()}_{key}"] = value
        hla_rows.append(row)
        if is_unresolved:
            gap_row = {
                "row_type": "UNRESOLVED_GAP_REFERENCE_ONLY",
                "date": date.isoformat(), "site": "HLA", "d222_rain_mm": "",
                "d222_value_status": d["value_status"], "previous_014_unresolved": True,
                "global_product_role": "REFERENCE_ONLY",
                "gpcc_reference_signal_mm": product_days["GPCC"]["HLA"][date]["nearest_grid_precip_mm"],
                "gpcc_reference_neighbor_max_mm": product_days["GPCC"]["HLA"][date]["neighbor_max_mm"],
                "cpc_reference_signal_mm": product_days["CPC"]["HLA"][date]["nearest_grid_precip_mm"],
                "cpc_reference_neighbor_max_mm": product_days["CPC"]["HLA"][date]["neighbor_max_mm"],
                "status": "REFERENCE_ONLY",
                "notes": "Gridded signal does not recover the station-day observation and is not installed into D222/weather.",
            }
            hla_rows.append(gap_row)

    for month in range(1, 13):
        dates = period_dates(2013, month)
        summary = site_numeric_summary("HLA", dates, d222)
        row = {"row_type": "MONTHLY_SUMMARY", "site": "HLA", "year": 2013, "month": month, **summary}
        for product in PRODUCTS:
            row.update({f"{product.lower()}_{key}": value for key, value in product_period_summary("HLA", product, dates, product_days, d222["HLA"]).items()})
        hla_rows.append(row)
    hla_dates = period_dates(2013)
    hla_summary = {"row_type": "ANNUAL_CONTEXT_NOT_D222_TOTAL", "site": "HLA", "year": 2013, **site_numeric_summary("HLA", hla_dates, d222)}
    for product in PRODUCTS:
        hla_summary.update({f"{product.lower()}_{key}": value for key, value in product_period_summary("HLA", product, hla_dates, product_days, d222["HLA"]).items()})
    hla_summary["notes"] = "HLA D222 is incomplete; D222 numeric subtotal is not an annual total. Global products provide regional context only."
    hla_rows.append(hla_summary)

    hla_metric_rows: list[dict[str, Any]] = []
    for product in PRODUCTS:
        dvals, pvals = [], []
        for date in hla_dates:
            dvalue = d222["HLA"].get(date, {}).get("rain_mm")
            pvalue = numeric_or_none(product_days[product]["HLA"].get(date, {}).get("nearest_grid_precip_mm"))
            if dvalue is not None and pvalue is not None:
                dvals.append(dvalue)
                pvals.append(pvalue)
        hla_metric_rows.append({"row_type": "D222_NUMERIC_DAY_METRICS", "site": "HLA", "year": 2013, "product": product, **pair_metrics(dvals, pvals), "bias_definition": "product minus D222 on same dates"})
    hla_rows.extend(hla_metric_rows)
    append_shift_diagnostics(hla_rows, "HLA", 2013, d222, product_days)

    # HLA key-event table rows.
    for date_text in EVENT_DATES["HLA"]:
        event = dt.date.fromisoformat(date_text)
        d = d222["HLA"].get(event, {"rain_mm": None, "value_status": "DATE_ABSENT"})
        row = {"row_type": "KEY_EVENT", "date": date_text, "site": "HLA", "d222_rain_mm": rounded(d["rain_mm"], 3), "d222_value_status": d["value_status"], "global_product_role": "REFERENCE_ONLY" if d["rain_mm"] is None else "REGIONAL_COMPARISON"}
        for product in PRODUCTS:
            row.update(product_event_summary("HLA", product, event, product_days))
        hla_rows.append(row)

    # FQA daily rows, monthly/annual summaries, unresolved context, and top events.
    for year in YEARS:
        for date in period_dates(year):
            d = d222["FQA"].get(date, {"rain_mm": None, "value_status": "DATE_ABSENT"})
            is_unresolved = date.isoformat() in unresolved["fqa_unresolved_dates"]
            row = {
                "row_type": "DAILY_2011_2013", "date": date.isoformat(), "site": "FQA",
                "year": year, "d222_rain_mm": rounded(d["rain_mm"], 3),
                "d222_value_status": d["value_status"], "previous_008_009_unresolved": is_unresolved,
                "global_product_role": "REFERENCE_ONLY" if is_unresolved else "REGIONAL_COMPARISON",
            }
            for product in PRODUCTS:
                item = site_product_row(product_days[product]["FQA"][date])
                for key, value in item.items():
                    if key not in ("date", "site", "product"):
                        row[f"{product.lower()}_{key}"] = value
            fqa_rows.append(row)

        year_dates = period_dates(year)
        row = {"row_type": "ANNUAL_SUMMARY", "site": "FQA", "year": year, **site_numeric_summary("FQA", year_dates, d222)}
        for product in PRODUCTS:
            row.update({f"{product.lower()}_{key}": value for key, value in product_period_summary("FQA", product, year_dates, product_days, d222["FQA"]).items()})
        row["notes"] = "D222 annual_total is blank unless all expected days have numeric values; its subtotal is not a complete-year total." 
        fqa_rows.append(row)

        for month in range(1, 13):
            dates = period_dates(year, month)
            month_row = {"row_type": "MONTHLY_SUMMARY", "site": "FQA", "year": year, "month": month, **site_numeric_summary("FQA", dates, d222)}
            for product in PRODUCTS:
                month_row.update({f"{product.lower()}_{key}": value for key, value in product_period_summary("FQA", product, dates, product_days, d222["FQA"]).items()})
            fqa_rows.append(month_row)

    fqa_numeric_events = []
    for date, value in d222["FQA"].items():
        if date.year in YEARS and value["rain_mm"] is not None:
            fqa_numeric_events.append((value["rain_mm"], date))
    fqa_numeric_events.sort(key=lambda item: (-item[0], item[1]))
    top20 = {date for _, date in fqa_numeric_events[:20]}
    selected = top20 | {dt.date.fromisoformat(value) for value in EVENT_DATES["FQA"]}
    rank = {date: index + 1 for index, (_, date) in enumerate(fqa_numeric_events[:20])}
    for event in sorted(selected):
        d = d222["FQA"].get(event, {"rain_mm": None, "value_status": "DATE_ABSENT"})
        row = {"row_type": "TOP20_OR_KEY_EVENT", "date": event.isoformat(), "site": "FQA", "year": event.year,
               "event_rank_among_top20": rank.get(event, ""), "d222_rain_mm": rounded(d["rain_mm"], 3),
               "d222_value_status": d["value_status"], "previous_008_009_unresolved": event.isoformat() in unresolved["fqa_unresolved_dates"]}
        for product in PRODUCTS:
            row.update(product_event_summary("FQA", product, event, product_days))
        fqa_rows.append(row)

    for product in PRODUCTS:
        dvals, pvals = [], []
        for year in YEARS:
            for date in period_dates(year):
                dvalue = d222["FQA"].get(date, {}).get("rain_mm")
                pvalue = numeric_or_none(product_days[product]["FQA"].get(date, {}).get("nearest_grid_precip_mm"))
                if dvalue is not None and pvalue is not None:
                    dvals.append(dvalue)
                    pvals.append(pvalue)
        fqa_rows.append({"row_type": "D222_NUMERIC_DAY_METRICS", "site": "FQA", "year": "2011-2013", "product": product, **pair_metrics(dvals, pvals), "bias_definition": "product minus D222 on same dates"})
    for year in YEARS:
        append_shift_diagnostics(fqa_rows, "FQA", year, d222, product_days)

    # Product-to-product daily, monthly, annual and event agreement.
    for site in ("HLA", "FQA"):
        all_dates = [date for year in YEARS for date in period_dates(year)]
        gpcc_values = [numeric_or_none(product_days["GPCC"][site][date]["nearest_grid_precip_mm"]) for date in all_dates]
        cpc_values = [numeric_or_none(product_days["CPC"][site][date]["nearest_grid_precip_mm"]) for date in all_dates]
        both = [(g, c) for g, c in zip(gpcc_values, cpc_values) if g is not None and c is not None]
        dmetrics = pair_metrics([pair[0] for pair in both], [pair[1] for pair in both])
        cross_rows.append({"row_type": "DAILY_NEAREST_GRID_AGREEMENT", "site": site, "period": "2011-2013", **dmetrics, "reference_product": "GPCC", "comparison_product": "CPC"})
        monthly_gpcc, monthly_cpc = [], []
        for year in YEARS:
            annual_dates = period_dates(year)
            gpcc_annual = product_period_summary(site, "GPCC", annual_dates, product_days, d222[site])
            cpc_annual = product_period_summary(site, "CPC", annual_dates, product_days, d222[site])
            gtotal = numeric_or_none(gpcc_annual["product_full_period_total_mm"])
            ctotal = numeric_or_none(cpc_annual["product_full_period_total_mm"])
            cross_rows.append({
                "row_type": "ANNUAL_NEAREST_GRID_TOTAL_RATIO", "site": site, "year": year,
                "gpcc_annual_total_mm": gpcc_annual["product_full_period_total_mm"],
                "cpc_annual_total_mm": cpc_annual["product_full_period_total_mm"],
                "gpcc_to_cpc_annual_total_ratio": rounded(gtotal / ctotal, 6) if gtotal is not None and ctotal else "",
            })
            for month in range(1, 13):
                dates = period_dates(year, month)
                gs = product_period_summary(site, "GPCC", dates, product_days, d222[site])
                cs = product_period_summary(site, "CPC", dates, product_days, d222[site])
                gv, cv = numeric_or_none(gs["product_full_period_total_mm"]), numeric_or_none(cs["product_full_period_total_mm"])
                if gv is not None and cv is not None:
                    monthly_gpcc.append(gv)
                    monthly_cpc.append(cv)
                cross_rows.append({
                    "row_type": "MONTHLY_NEAREST_GRID_TOTAL", "site": site, "year": year, "month": month,
                    "gpcc_monthly_total_mm": gs["product_full_period_total_mm"],
                    "cpc_monthly_total_mm": cs["product_full_period_total_mm"],
                    "gpcc_to_cpc_monthly_total_ratio": rounded(gv / cv, 6) if gv is not None and cv else "",
                })
        if monthly_gpcc and monthly_cpc:
            mm = pair_metrics(monthly_gpcc, monthly_cpc)
            cross_rows.append({"row_type": "MONTHLY_TOTAL_CORRELATION", "site": site, "period": "2011-2013", "monthly_overlap_count": len(monthly_gpcc), "pearson_r": mm["pearson_r"], "spearman_rho": mm["spearman_rho"]})
        event_dates = sorted({dt.date.fromisoformat(text) for text in EVENT_DATES[site]})
        if site == "FQA":
            event_dates = sorted(selected)
        both_same_day_strong = 0
        either_same_day_strong = 0
        either_pm2_strong = 0
        for event in event_dates:
            gp = product_event_summary(site, "GPCC", event, product_days)
            cp = product_event_summary(site, "CPC", event, product_days)
            gb = bool(gp["gpcc_strong_region_same_day_ge50"])
            cb = bool(cp["cpc_strong_region_same_day_ge50"])
            both_same_day_strong += gb and cb
            either_same_day_strong += gb or cb
            either_pm2_strong += bool(gp["gpcc_strong_region_pm2_ge50"] or cp["cpc_strong_region_pm2_ge50"])
            cross_rows.append({
                "row_type": "MAJOR_EVENT_PRODUCT_AGREEMENT", "site": site, "event_date": event.isoformat(),
                "d222_mm": rounded(d222[site].get(event, {}).get("rain_mm"), 3),
                "gpcc_nearest_mm": gp["gpcc_nearest_mm"], "cpc_nearest_mm": cp["cpc_nearest_mm"],
                "gpcc_3x3_max_pm2_mm": gp["gpcc_max_3x3_pm2_mm"], "cpc_3x3_max_pm2_mm": cp["cpc_max_3x3_pm2_mm"],
                "both_products_same_day_regional_ge50": gb and cb,
                "either_product_within_pm2_regional_ge50": bool(gp["gpcc_strong_region_pm2_ge50"] or cp["cpc_strong_region_pm2_ge50"]),
            })
        cross_rows.append({
            "row_type": "MAJOR_EVENT_AGREEMENT_SUMMARY", "site": site, "period": "2011-2013 selected events",
            "major_events_compared": len(event_dates), "both_products_same_day_regional_ge50_count": both_same_day_strong,
            "either_product_same_day_regional_ge50_count": either_same_day_strong,
            "either_product_pm2_regional_ge50_count": either_pm2_strong,
            "strong_rain_threshold_mm": STRONG_RAIN_MM,
        })
    fqa_annual_rows = [row for row in fqa_rows if row["row_type"] == "ANNUAL_SUMMARY"]
    for product in PRODUCTS:
        prefix = product.lower()
        totals = [numeric_or_none(row.get(f"{prefix}_product_full_period_total_mm")) for row in fqa_annual_rows]
        complete_totals = [value for value in totals if value is not None]
        three_year_mean = float(np.mean(complete_totals)) if complete_totals else None
        for row in fqa_annual_rows:
            product_total = numeric_or_none(row.get(f"{prefix}_product_full_period_total_mm"))
            row[f"{prefix}_annual_total_ratio_to_2011_2013_mean"] = rounded(product_total / three_year_mean, 6) if product_total is not None and three_year_mean else ""
            row[f"d222_numeric_subtotal_to_{prefix}_annual_grid_total_ratio"] = rounded(float(row["d222_numeric_subtotal_mm"]) / product_total, 6) if product_total and row.get("d222_numeric_subtotal_mm") not in (None, "") else ""
    return hla_rows, fqa_rows, cross_rows


def input_metadata_and_product_identity(site_coordinates: dict[str, Any], file_meta: dict[str, Any], gpcc_headers: dict[int, Any], gpcc_coords: tuple[np.ndarray, np.ndarray], cpc_meta: dict[int, Any]) -> tuple[dict[str, Any], str]:
    gpcc_lat, gpcc_lon = gpcc_coords
    gpcc_meta = grid_product_metadata(GPCC_FILES[2011], gpcc_headers[2011], gpcc_lat, gpcc_lon)
    # Reconcile the product metadata common to all three annual files.
    gpcc_meta["annual_file_record_counts"] = {str(year): gpcc_headers[year]["num_records"] for year in YEARS}
    gpcc_meta["annual_files"] = [file_meta[rel(GPCC_FILES[year])] for year in YEARS]
    cpc_common = cpc_meta[2011]
    cpc_common["annual_file_record_counts"] = {str(year): cpc_meta[year]["dimensions"]["time"] for year in YEARS}
    cpc_common["annual_files"] = [file_meta[rel(CPC_FILES[year])] for year in YEARS]
    products = {
        "GPCC_FULL_DATA_DAILY_V2022": {
            **gpcc_meta,
            "product_name": "GPCC Full Data Daily Product Version 2022, 1.0 degree",
            "gauge_based_gridded_product": True,
            "daily_resolution": True,
            "spatial_resolution_degrees": "1.0 x 1.0",
            "source_characterization": "Land-surface daily totals based on national meteorological/hydrological services, regional/global data collections, and WMO GTS; daily station relative anomalies interpolated and superimposed on the monthly GPCC product.",
            "quality_and_station_auxiliary_variables": ["interpolation_error (mm/day)", "numgauge (gauges per grid cell)"],
        },
        "CPC_GLOBAL_DAILY_PRECIP": {
            **cpc_common,
            "product_name": "CPC Unified Gauge-Based Analysis of Global Daily Precipitation, V1.0 RT for 2006-present",
            "gauge_based_gridded_product": True,
            "daily_resolution": True,
            "spatial_resolution_degrees": "0.5 x 0.5",
            "source_characterization": "Global land daily gauge analysis using CPC unified sources and objective analysis; 2006-present RT product.",
            "quality_and_station_auxiliary_variables": "No gauge-count or quality variable is present in the selected NetCDF precip files. The CPC README documents a gauge-count field in its separate GrADS distribution and warns that EOD information is limited and unchecked.",
        },
    }
    provenance = {
        "task": "023", "audit_only": True, "audit_date": "2026-10-07",
        "inputs": list(file_meta.values()),
        "gpcc_uncompressed_read_method": "Sequential gzip + CDF-1 record streaming; no decompressed copy was created; global arrays were not loaded.",
        "cpc_read_method": "h5py daily chunk hyperslabs; no full annual global data array was loaded.",
        "station_coordinates_source": {site: value for site, value in site_coordinates.items()},
        "d222_reference": file_meta[rel(D222_CSV)],
        "d222_snapshot_identity": json.loads(D222_IDENTITY.read_text(encoding="utf-8")),
        "unresolved_date_lists": {
            "HLA_source": file_meta[rel(HLA_UNRESOLVED_CSV)],
            "FQA_source": file_meta[rel(FQA_AUDIT_JSON)],
        },
        "products": products,
        "comparison_conventions": {
            "nearest_grid_is_primary": True,
            "3x3_is_spatial_context_only": True,
            "grid_selection_rule": "Geodesically nearest center to WTH coordinates; no value matching or best-fit grid selection.",
            "distance_formula": "great-circle haversine on spherical Earth radius 6371.0088 km",
            "strong_regional_rain_diagnostic_mm_day": STRONG_RAIN_MM,
            "timezone_status": "Neither NetCDF metadata nor source documentation supports relabeling both products as uniform UTC-day totals; no daily shift is applied in official outputs.",
        },
    }
    identity_lines = [
        "# GPCC 与 CPC 产品身份核查（Task 023）", "",
        "## GPCC Full Data Daily v2022", "",
        "本次文件头身份为 GPCC Full Data Daily Product Version 2022、1.0°、日尺度、陆地雨量计格点产品。文件说明其根据国家气象/水文部门、区域和全球资料集及 WMO GTS 雨量资料构建：将站点日/月份相对降水异常插值后叠加在 GPCC 月产品的月总量上。它不是海伦或封丘单站逐日观测。变量包含 `precip`（mm/day）、`interpolation_error`（mm/day）和 `numgauge`（gauges per grid cell）；缺测值为 -9999。三个本地年度文件分别覆盖 2011、2012、2013，CDF-1 / gzip；经纬度网格中心、变量属性、时间单位、SHA256 见 `input_provenance.json`。官方产品资料期截至 2020 年；产品页（本次查阅日期 2026-10-07）已列出 v2025，本审计仍按本地冻结的 v2022 文件执行：[GPCC Full Data Daily v2022](https://opendata.dwd.de/climate_environment/GPCC/html/fulldata-daily_v2022_doi_download.html)。",
        "",
        "## CPC Global Precipitation", "",
        "本次文件头身份为 `CPC GLOBAL PRCP V1.0 RT`，`precip(time,lat,lon)` 单位 mm、日总量，0.5°全球陆地区域网格。官方资料期自 1979 年开始，2011–2013 属 2006 年后的 real-time 产品时段；本地文件的变量/属性、时间轴和压缩块信息见 `input_provenance.json`。该 NetCDF 文件只含降水及坐标/时间变量，没有 gauge-count 或质量辅助变量。CPC README 说明分析基于统一雨量计来源与客观分析，空间分辨率 0.5°；还明确写明各国日累积结束时刻不同，所附 EOD 信息有限且未经充分核验：[CPC Global Daily README](https://ftp.cpc.ncep.noaa.gov/precip/CPC_UNI_PRCP/GAUGE_GLB/DOCU/PRCP_CU_GAUGE_V1.0GLB_0.50deg_README.txt)。NOAA PSL 也将该产品列为 0.5° 日降水格点产品：[NOAA PSL gridded precipitation catalog](https://psl.noaa.gov/data/gridded/tables/att.precipitation.html)。",
        "",
        "## 来源独立性与日界", "",
        "GPCC 与 CPC 都是格点产品，不是 HLA/FQA 本站观测，也不能视为完全独立观测。GPCC 明示输入包括国家资料和 GTS；CPC 的全球统一分析也整合多来源雨量站资料并使用 GTS 站报告，因此底层站源可能重叠。两产品相互一致只能说明区域产品之间的一致程度，不能替代单站观测核查。",
        "",
        "GPCC 时间坐标采用 `days since YYYY-1-1 00:00:00`、`proleptic_gregorian`；CPC 采用 `hours since 1900-01-01 00:00:00` 且坐标标为 `start`。NetCDF 中的 00:00 是日期标签，不足以证明统一 UTC 观测日界。CPC README 进一步指出日累积结束时刻因国家而异。输出保留原始日期；±1 日仅作为对齐敏感性诊断，未平移产品或 D222。",
        "",
        "## 网格提取", "",
        "每个站点、每个产品均按项目 WTH 经纬度选取球面距离最近的格点作为主分析，并同时报告周边 3×3 的均值、中位数、最小值、最大值。距离和格点中心随逐日产品 CSV 输出。3×3 仅作空间背景，未按与 D222 的相似度挑选格点。",
        "",
        "本地文件固定为 v2022 GPCC 与 V1.0 RT CPC。报告期间没有更新、复制、解压落地或改写任何原始文件。",
    ]
    return provenance, "\n".join(identity_lines) + "\n"


def build_site_coordinate_doc(site_coordinates: dict[str, Any], grids: dict[str, Any]) -> dict[str, Any]:
    result = {"coordinate_selection": "Project station coordinates are taken from the site WTH header, as allowed by Task 023.", "sites": {}}
    for site, coord in site_coordinates.items():
        result["sites"][site] = {**coord, "nearest_grids": grids[site]}
    if HLA_ALT_IDENTITY.is_file():
        alt = json.loads(HLA_ALT_IDENTITY.read_text(encoding="utf-8"))
        ghcn = alt.get("metadata_discrepancy", {}).get("ghcn_location")
        if ghcn and len(ghcn) >= 2:
            wth = site_coordinates["HLA"]
            result["sites"]["HLA"]["alternative_audited_station_location"] = {
                "latitude": float(ghcn[0]), "longitude": float(ghcn[1]),
                "source_path": rel(HLA_ALT_IDENTITY),
                "distance_from_project_wth_km": rounded(haversine_km(wth["latitude"], wth["longitude"], float(ghcn[0]), float(ghcn[1])), 3),
                "note": "NOAA/GHCN station metadata is recorded as an alternative; nearest grid extraction follows the project WTH position, not this alternative.",
            }
    return result


def main() -> None:
    if not OUT.is_dir():
        raise FileNotFoundError(f"Create the new output directory first: {OUT}")
    conflicts = [str(OUT / name) for name in OUTPUT_NAMES if (OUT / name).exists()]
    if DOC.exists():
        conflicts.append(str(DOC))
    if IDENTITY_DOC.exists():
        conflicts.append(str(IDENTITY_DOC))
    if conflicts:
        raise FileExistsError("Refusing to overwrite Task 023 audit outputs: " + ", ".join(conflicts))
    missing = [str(path) for path in [*GPCC_FILES.values(), *CPC_FILES.values(), D222_CSV, D222_IDENTITY, HLA_UNRESOLVED_CSV, FQA_AUDIT_JSON, *COORD_WTH.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required project inputs missing: " + ", ".join(missing))

    site_coordinates = {site: read_wth_coordinate(path, site) for site, path in COORD_WTH.items()}
    d222, _, unresolved = parse_d222_inputs()
    input_paths = {
        **{rel(path): "frozen GPCC daily input; read-only" for path in GPCC_FILES.values()},
        **{rel(path): "frozen CPC global daily input; read-only" for path in CPC_FILES.values()},
        rel(D222_CSV): "current D222 normalized audit reference from Task 022; read-only",
        rel(D222_IDENTITY): "Task 022 D222 snapshot identity; read-only",
        rel(COORD_WTH["HLA"]): "HLA project WTH coordinate header source; read-only",
        rel(COORD_WTH["FQA"]): "FQA project WTH coordinate header source; read-only",
        rel(HLA_UNRESOLVED_CSV): "frozen HLA unresolved date list after Task 014; read-only",
        rel(FQA_AUDIT_JSON): "frozen FQA unresolved date list/provenance audit; read-only",
    }
    if HLA_ALT_IDENTITY.is_file():
        input_paths[rel(HLA_ALT_IDENTITY)] = "alternative audited HLA station coordinates context; read-only"
    file_meta = {path: input_file_record(ROOT / path, role) for path, role in input_paths.items()}

    gpcc_headers = {year: parse_gpcc_cdf_header(path) for year, path in GPCC_FILES.items()}
    gpcc_lat = read_fixed_cdf_variable(GPCC_FILES[2011], gpcc_headers[2011]["variable_map"]["lat"]).ravel()
    gpcc_lon = read_fixed_cdf_variable(GPCC_FILES[2011], gpcc_headers[2011]["variable_map"]["lon"]).ravel()
    if any(gpcc_headers[year]["num_records"] != len(period_dates(year)) for year in YEARS):
        raise ValueError("GPCC yearly record count does not match calendar days")

    product_days: dict[str, dict[str, dict[dt.date, dict[str, Any]]]] = {product: {site: {} for site in site_coordinates} for product in PRODUCTS}
    cpc_metadata: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        print(f"Streaming GPCC {year}", flush=True)
        gpcc_year = gpcc_daily_extract(GPCC_FILES[year], year, site_coordinates, gpcc_headers[year])
        for site in site_coordinates:
            product_days["GPCC"][site].update(gpcc_year[site])
        del gpcc_year
        print(f"Reading CPC daily chunks {year}", flush=True)
        cpc_year, cpc_metadata[year] = cpc_daily_extract(CPC_FILES[year], year, site_coordinates)
        for site in site_coordinates:
            product_days["CPC"][site].update(cpc_year[site])
        del cpc_year

    grids = {}
    for site, coordinate in site_coordinates.items():
        grids[site] = {}
        for product in PRODUCTS:
            first = product_days[product][site][dt.date(YEARS[0], 1, 1)]
            grids[site][product] = {
                "grid_latitude": first["nearest_grid_latitude"],
                "grid_longitude": first["nearest_grid_longitude"],
                "nearest_grid_distance_km": first["nearest_grid_distance_km"],
            }
    coordinate_doc = build_site_coordinate_doc(site_coordinates, grids)
    (OUT / "site_coordinates.json").write_text(json.dumps(coordinate_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    for site in site_coordinates:
        for product in PRODUCTS:
            rows = [site_product_row(record) for _, record in sorted(product_days[product][site].items())]
            write_csv(OUT / f"{site.lower()}_{product.lower()}_daily_2011_2013.csv", rows)

    provenance, identity_markdown = input_metadata_and_product_identity(site_coordinates, file_meta, gpcc_headers, (gpcc_lat, gpcc_lon), cpc_metadata)
    provenance["unresolved_date_counts"] = {"HLA_previous_unresolved_days": len(unresolved["hla_unresolved_dates"]), "FQA_previous_unresolved_days": len(unresolved["fqa_unresolved_dates"])}
    provenance["unresolved_date_lists"]["HLA_count"] = len(unresolved["hla_unresolved_dates"])
    provenance["unresolved_date_lists"]["FQA_count"] = len(unresolved["fqa_unresolved_dates"])
    provenance["products"]["CPC_GLOBAL_DAILY_PRECIP"]["annual_file_record_counts"] = {str(year): cpc_metadata[year]["dimensions"]["time"] for year in YEARS}
    provenance["products"]["CPC_GLOBAL_DAILY_PRECIP"]["annual_files"] = [file_meta[rel(CPC_FILES[year])] for year in YEARS]
    (OUT / "input_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "product_identity.md").write_text(identity_markdown, encoding="utf-8")
    IDENTITY_DOC.parent.mkdir(parents=True, exist_ok=True)
    IDENTITY_DOC.write_text(identity_markdown, encoding="utf-8")

    hla_rows, fqa_rows, cross_rows = create_sanity_outputs(d222, product_days, unresolved)
    write_csv(OUT / "hla_2013_sanity.csv", hla_rows)
    write_csv(OUT / "fqa_2011_2013_sanity.csv", fqa_rows)
    write_csv(OUT / "gpcc_vs_cpc_comparison.csv", cross_rows)

    hla_gap_rows = [row for row in hla_rows if row["row_type"] == "UNRESOLVED_GAP_REFERENCE_ONLY"]
    hla_gap_signal = {}
    for product in PRODUCTS:
        prefix = product.lower()
        hla_gap_signal[product] = {
            "reference_only_days": len(hla_gap_rows),
            "nearest_wet_day_count": sum((numeric_or_none(row.get(f"{prefix}_reference_signal_mm")) or 0) > 0 for row in hla_gap_rows),
            "neighbor_3x3_ge_50mm_day_count": sum((numeric_or_none(row.get(f"{prefix}_reference_neighbor_max_mm")) or 0) >= STRONG_RAIN_MM for row in hla_gap_rows),
        }
    hla_events = [row for row in hla_rows if row["row_type"] == "KEY_EVENT"]
    fqa_events = [row for row in fqa_rows if row["row_type"] == "TOP20_OR_KEY_EVENT"]
    fqa_key_event_status = {}
    for date_text in EVENT_DATES["FQA"]:
        row = next(item for item in fqa_events if item["date"] == date_text)
        status = {}
        for product in PRODUCTS:
            p = product.lower()
            status[product] = {
                "same_day_3x3_ge_50": bool(row.get(f"{p}_strong_region_same_day_ge50")),
                "within_pm1_3x3_ge_50": bool(numeric_or_none(row.get(f"{p}_max_3x3_pm1_mm")) is not None and float(row[f"{p}_max_3x3_pm1_mm"]) >= STRONG_RAIN_MM),
                "within_pm2_3x3_ge_50": bool(numeric_or_none(row.get(f"{p}_max_3x3_pm2_mm")) is not None and float(row[f"{p}_max_3x3_pm2_mm"]) >= STRONG_RAIN_MM),
            }
        fqa_key_event_status[date_text] = status

    gate = {
        "task": "023", "audit_only": True, "audit_date": "2026-10-07",
        "products": ["GPCC_FULL_DATA_DAILY_V2022", "CPC_GLOBAL_DAILY_PRECIP"],
        "existing_data_modified": False, "existing_wth_modified": False, "existing_results_modified": False,
        "wgen_executed": False, "dssat_executed": False, "ppo_executed": False,
        "HLA": {
            "global_product_sanity_status": "REGIONAL_PRODUCTS_PARTIALLY_SUPPORT_D222",
            "unresolved_days_before": len(unresolved["hla_unresolved_dates"]),
            "unresolved_days_after": len(unresolved["hla_unresolved_dates"]),
            "unresolved_day_status": "REFERENCE_ONLY",
            "unresolved_day_product_signal_counts": hla_gap_signal,
            "global_product_role": "REFERENCE_ONLY", "gaps_filled": False,
            "d222_2013_numeric_day_count": site_numeric_summary("HLA", period_dates(2013), d222)["d222_numeric_days"],
            "key_event_count": len(hla_events),
        },
        "FQA": {
            "global_product_sanity_status": "MIXED_OR_INCONCLUSIVE",
            "extreme_event_support_rule": "Both products must show a 3x3 neighborhood maximum >= 50 mm/day on the event date or within +/- 2 days.",
            "extreme_event_2012_09_02_supported": all(fqa_key_event_status["2012-09-02"][product]["within_pm2_3x3_ge_50"] for product in PRODUCTS),
            "extreme_event_2013_05_26_supported": all(fqa_key_event_status["2013-05-26"][product]["within_pm2_3x3_ge_50"] for product in PRODUCTS),
            "key_event_regional_signal_ge_50mm": fqa_key_event_status,
            "global_product_role": "REFERENCE_ONLY", "gaps_filled": False,
            "weather_ready_for_wgen": False,
            "previous_unresolved_days": len(unresolved["fqa_unresolved_dates"]),
            "unresolved_day_status": "REFERENCE_ONLY",
        },
        "coordinate_selection": "Project WTH coordinates; current grid center selected solely by nearest geodesic distance.",
        "strong_regional_rain_diagnostic_mm_day": STRONG_RAIN_MM,
        "next_step": "REVIEW_RESULTS_THEN_DECIDE_SEPARATE_GRIDDED_FILL_SENSITIVITY_TASK",
    }
    (OUT / "final_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    hla_d = site_numeric_summary("HLA", period_dates(2013), d222)
    fqa_annual_rows = [row for row in fqa_rows if row["row_type"] == "ANNUAL_SUMMARY"]
    product_distances = {site: {product: grids[site][product] for product in PRODUCTS} for site in site_coordinates}
    hla_event_text = []
    for row in hla_events:
        hla_event_text.append(
            f"| {row['date']} | {row['d222_rain_mm'] or '缺测'} | {row['gpcc_nearest_mm']} | {row['gpcc_3x3_min_mm']}–{row['gpcc_3x3_max_mm']} | {row['cpc_nearest_mm']} | {row['cpc_3x3_min_mm']}–{row['cpc_3x3_max_mm']} | {row['gpcc_max_3x3_pm1_mm']} | {row['gpcc_max_3x3_pm2_mm']} | {row['cpc_max_3x3_pm1_mm']} | {row['cpc_max_3x3_pm2_mm']} |"
        )
    fqa_annual_text = []
    for row in fqa_annual_rows:
        fqa_annual_text.append(
            f"| {row['year']} | {row['d222_numeric_days']}/{row['expected_days']} | {row['d222_missing_days']} | {row['d222_numeric_subtotal_mm']} | {row['gpcc_product_full_period_total_mm']} | {row['cpc_product_full_period_total_mm']} | {row['gpcc_annual_total_ratio_to_2011_2013_mean']} | {row['cpc_annual_total_ratio_to_2011_2013_mean']} | {row['d222_numeric_subtotal_to_gpcc_annual_grid_total_ratio']} | {row['d222_numeric_subtotal_to_cpc_annual_grid_total_ratio']} |"
        )
    fqa_key_event_text = []
    for date_text in EVENT_DATES["FQA"]:
        row = next(item for item in fqa_rows if item["row_type"] == "TOP20_OR_KEY_EVENT" and item["date"] == date_text)
        both_products_signal = row["gpcc_strong_region_pm2_ge50"] and row["cpc_strong_region_pm2_ge50"]
        fqa_key_event_text.append(
            f"| {date_text} | {row['d222_rain_mm']} | {row['gpcc_nearest_mm']} | {row['gpcc_3x3_min_mm']}–{row['gpcc_3x3_max_mm']} | {row['gpcc_max_3x3_pm1_mm']} | {row['gpcc_max_3x3_pm2_mm']} | {row['cpc_nearest_mm']} | {row['cpc_3x3_min_mm']}–{row['cpc_3x3_max_mm']} | {row['cpc_max_3x3_pm1_mm']} | {row['cpc_max_3x3_pm2_mm']} | {'是' if both_products_signal else '否'} |"
        )
    report = f"""# HLA / FQA 全球逐日降水产品合理性审计（023）

## 执行结论

本任务只读比较项目当前 D222 与 GPCC Full Data Daily v2022、CPC Global Precipitation V1.0 RT 的区域格点信号。采用 WTH 站点坐标的最近格点作主分析，周边 3×3 只用于空间背景。产品不是本站观测，也不是完全独立的两份证据。缺测未填补；输出状态固定为 `REFERENCE_ONLY`。没有修改任何现有数据、WTH 或结果，没有运行 WGEN、DSSAT 或 PPO。

GPCC 格式为 gzip NetCDF3，单年解压约 284 MB。本审计按记录流式读取，仅保留当日目标格点/周边值，没有创建解压副本，也没有把全年全球数组载入内存。

## 产品身份、来源和日界

- **GPCC**：GPCC Full Data Daily v2022，1°、逐日、陆地雨量计格点分析。文件有 `precip`、`interpolation_error` 和 `numgauge`，分别是降水 mm/day、插值误差 mm/day、格点雨量计数。官方资料期截至 2020 年；官方产品页（本次查阅日期 2026-10-07）已列出 v2025，但本审计仍使用本地冻结的 v2022 文件。输入来源及日异常插值方法见[GPCC v2022 产品说明](https://opendata.dwd.de/climate_environment/GPCC/html/fulldata-daily_v2022_doi_download.html)。
- **CPC**：CPC Unified Gauge-Based Analysis of Global Daily Precipitation，NetCDF 标记 `CPC GLOBAL PRCP V1.0 RT`，0.5°，逐日，单位 mm，规则全球陆地网格。官方资料期自 1979 年开始，2011–2013 属 2006 年后的 RT 时段。所给 NetCDF 只有降水/坐标/时间变量，没有格点雨量计数或质量变量。CPC README 警告不同国家日累积结束时刻不同且 EOD 辅助信息有限、未经核验。[CPC 产品 README](https://ftp.cpc.ncep.noaa.gov/precip/CPC_UNI_PRCP/GAUGE_GLB/DOCU/PRCP_CU_GAUGE_V1.0GLB_0.50deg_README.txt)

两个产品的纬度/经度变量表示规则网格中心；本地文件中的坐标、单位、缺测值、时间轴与维度定义见 `input_provenance.json`。产品身份、来源独立性和日界详情另见 [`hl_fq_global_precip_product_identity_023.md`](hl_fq_global_precip_product_identity_023.md)。

GPCC 的底层站资料与 CPC 的全球站网/GTS 来源可能部分重叠，因此两者不能称作完全独立观测。两个文件时间坐标均标为日开始日期/00:00，但都没有证明产品统一代表 UTC 日。CPC 文档明确指出日界随国家变化。本审计保留原日期；D222 对照的 −1/0/+1 日偏移仅是敏感性诊断，不改写任何数据。

六个年度文件的绝对路径、大小、mtime、SHA256、NetCDF 维度/变量/单位/缺测、时间轴和网格信息列在 `results/hl_fq_global_precip_023/input_provenance.json`；产品说明另见 `results/hl_fq_global_precip_023/product_identity.md` 和 [`hl_fq_global_precip_product_identity_023.md`](hl_fq_global_precip_product_identity_023.md)。

## 站点坐标和最近格点距离

站点坐标直接从项目 WTH 头部读取：HLA {site_coordinates['HLA']['latitude']:.3f}°N, {site_coordinates['HLA']['longitude']:.3f}°E；FQA {site_coordinates['FQA']['latitude']:.3f}°N, {site_coordinates['FQA']['longitude']:.3f}°E。HLA 的 NOAA/GHCN 审计坐标另有约 {coordinate_doc['sites']['HLA'].get('alternative_audited_station_location', {}).get('distance_from_project_wth_km', '')} km 差异，主分析仍依项目 WTH 坐标，选择规则没有按降水匹配结果改变。

| 站点 | 产品 | 最近格点中心 | 到 WTH 坐标距离 |
|---|---|---:|---:|
| HLA | GPCC | {product_distances['HLA']['GPCC']['grid_latitude']:.2f}°, {product_distances['HLA']['GPCC']['grid_longitude']:.2f}° | {product_distances['HLA']['GPCC']['nearest_grid_distance_km']:.2f} km |
| HLA | CPC | {product_distances['HLA']['CPC']['grid_latitude']:.2f}°, {product_distances['HLA']['CPC']['grid_longitude']:.2f}° | {product_distances['HLA']['CPC']['nearest_grid_distance_km']:.2f} km |
| FQA | GPCC | {product_distances['FQA']['GPCC']['grid_latitude']:.2f}°, {product_distances['FQA']['GPCC']['grid_longitude']:.2f}° | {product_distances['FQA']['GPCC']['nearest_grid_distance_km']:.2f} km |
| FQA | CPC | {product_distances['FQA']['CPC']['grid_latitude']:.2f}°, {product_distances['FQA']['CPC']['grid_longitude']:.2f}° | {product_distances['FQA']['CPC']['nearest_grid_distance_km']:.2f} km |

## HLA 2013

2013 年 D222 有 {hla_d['d222_numeric_days']} 个明确数值日、{hla_d['d222_missing_days']} 个缺测日；明示值小计 {hla_d['d222_numeric_subtotal_mm']} mm。由于覆盖不完整，该小计不是年总量。逐日对照仅对 D222 明确数值日计算 wet/dry、MAE、RMSE、Pearson、Spearman 和偏差，结果及时间偏移诊断见 `hla_2013_sanity.csv`。

| 日期 | D222 (mm) | GPCC nearest | GPCC 3×3 min–max | CPC nearest | CPC 3×3 min–max | GPCC ±1日3×3最大 | GPCC ±2日3×3最大 | CPC ±1日3×3最大 | CPC ±2日3×3最大 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(hla_event_text)}

014/021 的 111 个 unresolved 日期在审计表中逐日保留，并标为 `REFERENCE_ONLY`。GPCC 区域信号：{hla_gap_signal['GPCC']['nearest_wet_day_count']}/111 个最近格点有雨，{hla_gap_signal['GPCC']['neighbor_3x3_ge_50mm_day_count']}/111 天的 3×3 最大值达到 {STRONG_RAIN_MM:.0f} mm；CPC：{hla_gap_signal['CPC']['nearest_wet_day_count']}/111 个最近格点有雨，{hla_gap_signal['CPC']['neighbor_3x3_ge_50mm_day_count']}/111 天的 3×3 最大值达到该阈值。它们提供的是区域参考信号，不恢复 111 个本站日值。

GPCC 与 CPC 的 2013 最近格点年累计分别为 797.36 mm 和 760.885 mm，均高于本审计 2011–2012 对应年累计，表明 2013 在三年观察窗内区域偏湿，但不能代替长期气候异常检验。关键事件信号不一致：2013-08-12 两产品同日 3×3 均超过 50 mm/day；2013-07-30 只有 GPCC 达到该阈值；2013-08-09 两者均未达到。2013-07-21 只有 GPCC 在 ±2 日邻域达到阈值；2013-08-21 的 D222 空白日两者都没有强区域信号。

HLA 状态：`{gate['HLA']['global_product_sanity_status']}`。123 个明确数值日上，D222 对 GPCC/CPC 的 Pearson r 分别为 0.800/0.662；格点空间平均和事件幅度仍限制单日量级验证。区域数据部分支持湿事件及时间变化，但不足以把 178.2 mm 等站点量级全面判作已验证。

## FQA 2011–2013

| 年 | D222 数值日/应有日 | D222 缺测日 | D222 numeric subtotal (mm) | GPCC 年格点量 (mm) | CPC 年格点量 (mm) | GPCC/三年均值 | CPC/三年均值 | D222 小计/GPCC 年量 | D222 小计/CPC 年量 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(fqa_annual_text)}

D222 缺测时不报告完整年总量。两产品年量相对 2011–2013 三年均值的比值是短窗 anomaly ratio，不能代替多年气候态：2012、2013 在两产品上均低于该短窗均值。D222 每年的明示数值小计分别缺 314、335、330 天，不能当成完整年总量。虽然数值小计约为对应格点完整年量的 2.50–3.01 倍，但覆盖不全使两者不可直接按年总量比较。逐月 completeness、D222 小计、产品总量和共同日对照均在 `fqa_2011_2013_sanity.csv`。

2012-09-02 的 D222 240 mm 和 2013-05-26 的 204 mm，以及 D222 numeric subtotal 最大前 20 个事件均列入 FQA 表。区域强雨阈值为 50 mm/day；下表列出最近格点、当日 3×3 范围、±1/±2 日 3×3 最大值。“两产品区域支持”要求两个产品均在当日或 ±2 日内达到阈值，这仅表示区域降水信号，不验证站点量级。

| 日期 | D222 | GPCC 最近 | GPCC 当日3×3范围 | GPCC ±1最大 | GPCC ±2最大 | CPC 最近 | CPC 当日3×3范围 | CPC ±1最大 | CPC ±2最大 | 两产品区域支持 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(fqa_key_event_text)}

2012-09-02 的 GPCC/CPC ±2 日 3×3 最大值分别只有 27.41/11.126 mm/day，无同日或邻近强区域降水。2013-05-26 两产品当日 3×3 最大值为 65.85/60.566 mm/day，±2 日最大值为 77.14/60.676 mm/day；但最近格点只有 22.07/37.318 mm/day，未支持 D222 204 mm 的量级。FQA 年度 D222 小计覆盖极不完整，无法验证年总量，两个重点事件又是一有区域背景、一无区域强雨，因此整体判为混合/不确定。

FQA 状态：`{gate['FQA']['global_product_sanity_status']}`。当前 157 个 unresolved 日期不填补，天气仍不具备进入 WGEN 的条件。

## GPCC 与 CPC 一致性

`gpcc_vs_cpc_comparison.csv` 给出 HLA/FQA 的逐日最近格点 Pearson/Spearman、两种湿日阈值一致率、月总量相关、逐年总量比例和重点事件区域强雨一致情况。全 1096 日上，HLA/FQA 的 GPCC-CPC 日 Pearson r 分别为 0.933/0.872，Spearman rho 为 0.855/0.924，wet/dry（RAIN>0）一致率为 0.880/0.949；月总量 Pearson r 分别为 0.981/0.976。两区域产品整体同步，但潜在站源重叠、分辨率和日界差异意味着一致性不能证明本站值。

## 解释、后续与边界

这些格点值不能直接替换 D222：它们代表 0.5°/1°区域分析而非本站实测，CPC 日界依国家而异，GPCC/CPC 的站源可能重叠，格点会平滑点尺度极端值。HLA 111 天和 FQA 157 天均保持缺测；D222、cleaned weather、WTH、历史实验和结果都没有更改。

值得另行设计一项**格点产品辅助填补敏感性方案**，但本审计不足以授权或证明任何单日替换值。后续仅在隔离副本中比较“未填补原始版本”和明确标记的格点参考情景，并量化降水事件、年量及后续结论的变化；不得把格点值宣称为恢复的本站观测。本任务没有执行填补，也没有进入 WGEN/DSSAT/PPO。

## 输出

- 数据身份与坐标：`results/hl_fq_global_precip_023/input_provenance.json`、`site_coordinates.json`、`product_identity.md`、`docs/hl_fq_global_precip_product_identity_023.md`
- 四份逐日提取：`hla_gpcc_daily_2011_2013.csv`、`hla_cpc_daily_2011_2013.csv`、`fqa_gpcc_daily_2011_2013.csv`、`fqa_cpc_daily_2011_2013.csv`
- 专项 sanity 与产品一致性：`hla_2013_sanity.csv`、`fqa_2011_2013_sanity.csv`、`gpcc_vs_cpc_comparison.csv`
- 机器门禁：`final_gate.json`

本任务到此停止。是否设计后续隔离的填补敏感性方案，由用户查看本审计结果后另行决定。
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": "AUDIT_COMPLETE",
        "hla_unresolved_reference_only_rows": len(hla_gap_rows),
        "fqa_previous_unresolved_days": len(unresolved["fqa_unresolved_dates"]),
        "product_daily_rows_each": sum(len(product_days[product][site]) for product in PRODUCTS for site in site_coordinates),
        "outputs": [rel(OUT / name) for name in OUTPUT_NAMES] + [rel(DOC), rel(IDENTITY_DOC)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
