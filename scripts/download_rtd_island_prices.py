#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from rtd_download_core import (
    USER_AGENT,
    decode_md_file_param,
    encode_md_file_param,
    log,
    normalize_data_row,
    parse_timestamp,
)

DATASET_CODE = "RTD_ISLAND_PRICE"
DEFAULT_PAGE_URL = "https://www.iemop.ph/market-data/rtd-prices-and-schedules/"
DEFAULT_OUTPUT_ROOT = "data/rtd_island_prices"
DEFAULT_MD_FILE_PREFIX = "/var/www/html/wp-content/uploads/downloads/data/RTD/RTD_"
DEFAULT_MD_FILE_SUFFIX = ".zip"
RAW_HEADER = (
    "RUN_TIME",
    "MKT_TYPE",
    "TIME_INTERVAL",
    "REGION_NAME",
    "RESOURCE_NAME",
    "RESOURCE_TYPE",
    "SCHED_MW",
    "LMP",
    "LOSS_FACTOR",
    "LMP_SMP",
    "LMP_LOSS",
    "LMP_CONGESTION",
    "",
)
NORMALIZED_HEADER = RAW_HEADER[:-1]
NUMERIC_COLUMNS = {"SCHED_MW", "LMP", "LOSS_FACTOR", "LMP_SMP", "LMP_LOSS", "LMP_CONGESTION"}
CSV_EXPORT_DIR = Path("data/csv_exports_flat")


@dataclass(frozen=True)
class HourlyUrlPattern:
    page_url: str
    md_file_prefix: str
    md_file_suffix: str


@dataclass
class FileResult:
    file_token: str
    status: str
    url: str
    zip_path: str
    csv_path: str
    downloaded: bool
    http_status: int
    attachment_name: str
    zip_member_name: str
    bytes_downloaded: int
    raw_row_count: int
    data_row_count: int
    output_row_count: int
    interval_count: int
    region_names: str
    min_interval: str
    max_interval: str
    warnings: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download hourly IEMOP RTD Prices and Schedules ZIPs from a starting URL, "
            "walk forward until the latest available hourly file, compute schedule-weighted "
            "island prices by TIME_INTERVAL and REGION_NAME, and write one combined output."
        )
    )
    parser.add_argument(
        "--start-url",
        required=True,
        help="First IEMOP RTD Prices and Schedules download URL to process.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for raw downloads, QC manifest, and combined output.",
    )
    parser.add_argument(
        "--end-url",
        help=(
            "Optional last hourly IEMOP RTD Prices and Schedules download URL to process. "
            "When omitted, the script walks forward until the first unavailable hour."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers when --end-url is supplied. Defaults to 8.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds. Defaults to 60.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Download retries per file. Defaults to 3.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the starting file if a local ZIP/CSV already exists.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.timeout < 1:
        raise SystemExit("--timeout must be at least 1.")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1.")
    return args


def extract_hour_token_from_path(md_file_path: str) -> str:
    filename = Path(md_file_path).name
    stem = Path(filename).stem
    try:
        token = stem.rsplit("_", 1)[1]
    except IndexError as exc:
        raise ValueError(f"Could not extract hourly token from {filename!r}.") from exc
    if len(token) != 12 or not token.isdigit():
        raise ValueError(f"Expected YYYYMMDDHHMM hourly token in {filename!r}.")
    return token


def parse_start_url(start_url: str) -> tuple[HourlyUrlPattern, datetime]:
    parsed = urllib.parse.urlparse(start_url)
    query = urllib.parse.parse_qs(parsed.query)
    try:
        md_file_path = decode_md_file_param(query["md_file"][0])
    except (KeyError, IndexError) as exc:
        raise ValueError("Start URL must include exactly one md_file query parameter.") from exc
    token = extract_hour_token_from_path(md_file_path)
    suffix = Path(md_file_path).suffix
    prefix = md_file_path[: -len(token + suffix)]
    page_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    return (
        HourlyUrlPattern(
            page_url=page_url or DEFAULT_PAGE_URL,
            md_file_prefix=prefix or DEFAULT_MD_FILE_PREFIX,
            md_file_suffix=suffix or DEFAULT_MD_FILE_SUFFIX,
        ),
        datetime.strptime(token, "%Y%m%d%H%M"),
    )


def parse_end_url(end_url: str) -> tuple[HourlyUrlPattern, datetime]:
    return parse_start_url(end_url)


def build_url(pattern: HourlyUrlPattern, current_dt: datetime) -> str:
    md_file_path = (
        f"{pattern.md_file_prefix}{current_dt.strftime('%Y%m%d%H%M')}{pattern.md_file_suffix}"
    )
    md_file = encode_md_file_param(md_file_path)
    return f"{pattern.page_url}?{urllib.parse.urlencode({'md_file': md_file})}"


def request_download(url: str, timeout: int) -> tuple[bytes, int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        status = getattr(response, "status", 200)
        attachment_name = response.headers.get_filename() or ""
        return payload, status, attachment_name


def download_to_path(
    *,
    url: str,
    destination: Path,
    timeout: int,
    retries: int,
    force: bool,
) -> tuple[bool, int, str, int]:
    if destination.exists() and not force:
        return False, 0, destination.name, destination.stat().st_size

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            payload, http_status, attachment_name = request_download(url, timeout)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp_path = destination.with_suffix(destination.suffix + ".part")
            temp_path.write_bytes(payload)
            temp_path.replace(destination)
            return True, http_status, attachment_name, len(payload)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    assert last_error is not None
    raise last_error


def extract_single_csv(zip_path: Path, csv_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        members = [member for member in archive.namelist() if not member.endswith("/")]
        if len(members) != 1:
            raise ValueError(
                f"Expected exactly one file inside {zip_path.name}, found {len(members)}."
            )
        member_name = members[0]
        if Path(member_name).suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV inside {zip_path.name}, found {member_name!r}.")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member_name) as source, csv_path.open("wb") as dest:
            dest.write(source.read())
        return member_name


def validate_and_aggregate(csv_path: Path, attachment_name: str, zip_member_name: str) -> tuple[list[dict[str, object]], int, int, int, str, str, str, str]:
    warnings: list[str] = []
    raw_row_count = 0
    data_row_count = 0
    interval_values: set[datetime] = set()
    region_names: set[str] = set()
    mkt_types: set[str] = set()
    weight_sums: dict[tuple[datetime, str], float] = {}
    weighted_price_sums: dict[tuple[datetime, str], float] = {}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
            raw_row_count += 1
        except StopIteration as exc:
            raise ValueError("CSV is empty.") from exc

        if header == list(NORMALIZED_HEADER):
            warnings.append("header_missing_trailing_blank_column")
        elif header != list(RAW_HEADER):
            raise ValueError(f"Unexpected header: {header!r}")

        eof_ok = False
        for raw_row in reader:
            raw_row_count += 1
            if not raw_row:
                continue
            if raw_row == ["EOF"]:
                eof_ok = True
                break

            normalized_row = normalize_data_row(raw_row, _dataset_config_proxy(), warnings)
            row_map = dict(zip(NORMALIZED_HEADER, normalized_row))

            mkt_type = str(row_map["MKT_TYPE"])
            mkt_types.add(mkt_type)
            if mkt_type != "RTD":
                raise ValueError(f"Unexpected MKT_TYPE values: {sorted(mkt_types)!r}")

            try:
                interval_value = parse_timestamp(str(row_map["TIME_INTERVAL"]))
            except ValueError as exc:
                raise ValueError(f"Unsupported TIME_INTERVAL {row_map['TIME_INTERVAL']!r}") from exc

            for column in NUMERIC_COLUMNS:
                value = str(row_map[column])
                if value == "":
                    raise ValueError(f"Blank numeric value in {column}.")
                try:
                    row_map[column] = float(value)
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric value for {column}: {value!r}") from exc

            sched_mw = float(row_map["SCHED_MW"])
            lmp = float(row_map["LMP"])
            region_name = str(row_map["REGION_NAME"])
            key = (interval_value, region_name)
            weight = abs(sched_mw)
            weight_sums[key] = weight_sums.get(key, 0.0) + weight
            weighted_price_sums[key] = weighted_price_sums.get(key, 0.0) + (weight * lmp)
            interval_values.add(interval_value)
            region_names.add(region_name)
            data_row_count += 1

        if not eof_ok:
            warnings.append("missing_eof_marker")

    if attachment_name and attachment_name != csv_path.with_suffix(".zip").name:
        warnings.append(f"attachment_name_mismatch:{attachment_name}")
    if Path(zip_member_name).name != csv_path.name:
        warnings.append(f"zip_member_name_mismatch:{zip_member_name}")

    if data_row_count == 0:
        warnings.append("empty_data_file")
        return (
            [],
            raw_row_count,
            data_row_count,
            0,
            "",
            "",
            "",
            "|".join(dict.fromkeys(warnings)),
        )

    zero_weight_groups = [
        f"{interval_value.isoformat(sep=' ')}|{region_name}"
        for (interval_value, region_name), weight_sum in sorted(weight_sums.items())
        if weight_sum == 0.0
    ]
    if zero_weight_groups:
        preview = ", ".join(zero_weight_groups[:10])
        suffix = "" if len(zero_weight_groups) <= 10 else f" ... (+{len(zero_weight_groups) - 10} more)"
        raise ValueError(f"Zero island weight for interval-region group(s): {preview}{suffix}")

    output_rows = [
        {
            "TIME_INTERVAL": interval_value,
            "REGION_NAME": region_name,
            "ISLAND_PRICE": weighted_price_sums[(interval_value, region_name)]
            / weight_sums[(interval_value, region_name)],
            "WEIGHT_SUM": weight_sums[(interval_value, region_name)],
        }
        for interval_value, region_name in sorted(weight_sums)
    ]
    return (
        output_rows,
        raw_row_count,
        data_row_count,
        len(interval_values),
        "|".join(sorted(region_names)),
        min(interval_values).isoformat(sep=" "),
        max(interval_values).isoformat(sep=" "),
        "|".join(dict.fromkeys(warnings)),
    )


def _dataset_config_proxy():
    class ConfigProxy:
        raw_header = RAW_HEADER
        normalized_header = NORMALIZED_HEADER
        raw_column_count = len(RAW_HEADER)
        normalized_column_count = len(NORMALIZED_HEADER)

    return ConfigProxy()


def write_manifest(results: list[FileResult], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_outputs(rows: list[dict[str, object]], parquet_path: Path, csv_export_path: Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            {
                "TIME_INTERVAL": pd.Series(dtype="datetime64[ns]"),
                "REGION_NAME": pd.Series(dtype="category"),
                "ISLAND_PRICE": pd.Series(dtype="float64"),
                "WEIGHT_SUM": pd.Series(dtype="float64"),
            }
        )
    else:
        frame["TIME_INTERVAL"] = pd.to_datetime(frame["TIME_INTERVAL"], errors="raise")
        frame["REGION_NAME"] = frame["REGION_NAME"].astype("category")
        frame["ISLAND_PRICE"] = pd.to_numeric(frame["ISLAND_PRICE"], errors="raise").astype("float64")
        frame["WEIGHT_SUM"] = pd.to_numeric(frame["WEIGHT_SUM"], errors="raise").astype("float64")
        frame = frame.sort_values(["TIME_INTERVAL", "REGION_NAME"], kind="stable").reset_index(drop=True)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_export_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False, compression="zstd", engine="pyarrow")
    export_frame = frame.copy()
    export_frame["TIME_INTERVAL"] = export_frame["TIME_INTERVAL"].dt.strftime("%Y-%m-%d %H:%M:%S")
    export_frame.to_csv(csv_export_path, index=False)


def cleanup_success_files(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def iter_hour_tokens(start_dt: datetime, end_dt: datetime) -> list[datetime]:
    items: list[datetime] = []
    current_dt = start_dt
    while current_dt <= end_dt:
        items.append(current_dt)
        current_dt += timedelta(hours=1)
    return items


def process_fixed_range(
    *,
    start_dt: datetime,
    end_dt: datetime,
    pattern: HourlyUrlPattern,
    raw_dir: Path,
    timeout: int,
    retries: int,
    force: bool,
    workers: int,
) -> tuple[list[FileResult], list[dict[str, object]], list[Path]]:
    work_items = iter_hour_tokens(start_dt, end_dt)
    log(
        f"Processing fixed hourly range with {len(work_items)} files from "
        f"{start_dt.strftime('%Y-%m-%d %H:%M')} through {end_dt.strftime('%Y-%m-%d %H:%M')} "
        f"using {workers} workers"
    )

    results: list[FileResult] = []
    combined_rows: list[dict[str, object]] = []
    cleanup_paths: list[Path] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                process_hourly_file,
                current_dt=current_dt,
                pattern=pattern,
                raw_dir=raw_dir,
                timeout=timeout,
                retries=retries,
                force=force,
            ): current_dt
            for current_dt in work_items
        }
        for completed_count, future in enumerate(as_completed(future_map), start=1):
            current_dt = future_map[future]
            token = current_dt.strftime("%Y%m%d%H%M")
            log(f"[{DATASET_CODE} {token}] stage starting")
            result, output_rows, file_cleanup_paths = future.result()
            results.append(result)
            combined_rows.extend(output_rows)
            cleanup_paths.extend(file_cleanup_paths)
            warning_suffix = f" warnings={result.warnings}" if result.warnings else ""
            log(
                f"[{DATASET_CODE} {token}] QC {result.status} "
                f"data_rows={result.data_row_count} output_rows={result.output_row_count}{warning_suffix}"
            )
            log(f"Progress {completed_count}/{len(work_items)}")

    results.sort(key=lambda item: item.file_token)
    combined_rows.sort(key=lambda row: (row["TIME_INTERVAL"], row["REGION_NAME"]))
    deduped_cleanup_paths = list(dict.fromkeys(cleanup_paths))
    return results, combined_rows, deduped_cleanup_paths


def process_hourly_file(
    *,
    current_dt: datetime,
    pattern: HourlyUrlPattern,
    raw_dir: Path,
    timeout: int,
    retries: int,
    force: bool,
) -> tuple[FileResult, list[dict[str, object]], list[Path]]:
    token = current_dt.strftime("%Y%m%d%H%M")
    url = build_url(pattern, current_dt)
    zip_path = raw_dir / f"RTD_{token}.zip"
    csv_path = raw_dir / f"RTD_{token}.csv"
    downloaded = False
    http_status = 0
    attachment_name = ""
    bytes_downloaded = zip_path.stat().st_size if zip_path.exists() else 0

    if csv_path.exists() and not force:
        zip_member_name = csv_path.name
    else:
        downloaded, http_status, attachment_name, bytes_downloaded = download_to_path(
            url=url,
            destination=zip_path,
            timeout=timeout,
            retries=retries,
            force=force,
        )
        if downloaded:
            log(
                f"[{DATASET_CODE} {token}] download complete "
                f"http={http_status} bytes={bytes_downloaded}; extracting CSV"
            )
        else:
            log(f"[{DATASET_CODE} {token}] using existing ZIP; extracting CSV")
        zip_member_name = extract_single_csv(zip_path, csv_path)

    (
        output_rows,
        raw_row_count,
        data_row_count,
        interval_count,
        region_names,
        min_interval,
        max_interval,
        warnings,
    ) = validate_and_aggregate(csv_path, attachment_name, zip_member_name)

    result = FileResult(
        file_token=token,
        status="ok" if not warnings else "warning",
        url=url,
        zip_path=str(zip_path),
        csv_path=str(csv_path),
        downloaded=downloaded,
        http_status=http_status,
        attachment_name=attachment_name or zip_path.name,
        zip_member_name=zip_member_name,
        bytes_downloaded=bytes_downloaded,
        raw_row_count=raw_row_count,
        data_row_count=data_row_count,
        output_row_count=len(output_rows),
        interval_count=interval_count,
        region_names=region_names,
        min_interval=min_interval,
        max_interval=max_interval,
        warnings=warnings,
        error="",
    )
    cleanup_paths = [path for path in (zip_path, csv_path) if path.exists()] if result.status == "ok" else []
    return result, output_rows, cleanup_paths


def run() -> int:
    args = parse_args()
    pattern, start_dt = parse_start_url(args.start_url)
    end_dt: Optional[datetime] = None
    if args.end_url:
        end_pattern, end_dt = parse_end_url(args.end_url)
        if end_pattern.page_url != pattern.page_url:
            raise SystemExit("Start and end URLs must use the same download page.")
        if end_pattern.md_file_prefix != pattern.md_file_prefix:
            raise SystemExit("Start and end URLs must point to the same RTD ZIP pattern.")
        if end_pattern.md_file_suffix != pattern.md_file_suffix:
            raise SystemExit("Start and end URLs must share the same file suffix.")
        if end_dt < start_dt:
            raise SystemExit("End URL must not be earlier than the start URL.")
    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    qc_dir = output_root / "qc"
    combined_dir = output_root / "combined"

    results: list[FileResult] = []
    combined_rows: list[dict[str, object]] = []
    cleanup_paths: list[Path] = []
    latest_dt: Optional[datetime] = None

    if end_dt is not None:
        latest_dt = end_dt
        results, combined_rows, cleanup_paths = process_fixed_range(
            start_dt=start_dt,
            end_dt=end_dt,
            pattern=pattern,
            raw_dir=raw_dir,
            timeout=args.timeout,
            retries=args.retries,
            force=args.force,
            workers=args.workers,
        )
    else:
        current_dt = start_dt
        log(
            f"Starting {DATASET_CODE} run from {start_dt.strftime('%Y-%m-%d %H:%M')} "
            f"until latest available hourly ZIP"
        )

        while True:
            token = current_dt.strftime("%Y%m%d%H%M")
            log(f"[{DATASET_CODE} {token}] stage starting")
            try:
                result, output_rows, file_cleanup_paths = process_hourly_file(
                    current_dt=current_dt,
                    pattern=pattern,
                    raw_dir=raw_dir,
                    timeout=args.timeout,
                    retries=args.retries,
                    force=args.force,
                )
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and latest_dt is not None:
                    log(
                        f"[{DATASET_CODE} {token}] latest available file reached at "
                        f"{latest_dt.strftime('%Y-%m-%d %H:%M')}; stopping discovery"
                    )
                    break
                raise

            results.append(result)
            combined_rows.extend(output_rows)
            cleanup_paths.extend(file_cleanup_paths)
            latest_dt = current_dt
            warning_suffix = f" warnings={result.warnings}" if result.warnings else ""
            log(
                f"[{DATASET_CODE} {token}] QC {result.status} "
                f"data_rows={result.data_row_count} output_rows={result.output_row_count}{warning_suffix}"
            )
            current_dt += timedelta(hours=1)

    if latest_dt is None or not results:
        raise SystemExit("No files were processed.")

    start_token = start_dt.strftime("%Y%m%d%H%M")
    end_token = latest_dt.strftime("%Y%m%d%H%M")
    manifest_path = qc_dir / f"rtd_island_price_qc_{start_token}_{end_token}.csv"
    parquet_path = combined_dir / f"{DATASET_CODE}_{start_token}_{end_token}.parquet"
    csv_export_path = CSV_EXPORT_DIR / f"{DATASET_CODE}_{start_token}_{end_token}.csv"

    log("Writing QC manifest")
    write_manifest(results, manifest_path)
    log("Writing combined outputs")
    write_outputs(combined_rows, parquet_path, csv_export_path)
    log("Deleting successful raw ZIP/CSV files")
    cleanup_success_files(cleanup_paths)

    total_output_rows = sum(result.output_row_count for result in results)
    log(
        f"files={len(results)} warning={sum(result.status == 'warning' for result in results)} "
        f"output_rows={total_output_rows}"
    )
    log(f"QC manifest: {manifest_path}")
    log(f"Combined parquet: {parquet_path}")
    log(f"Flat CSV export: {csv_export_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
