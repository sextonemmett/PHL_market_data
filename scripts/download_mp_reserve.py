#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

BASE_DOWNLOAD_PAGE = "https://www.iemop.ph/market-data/rtd-reserve-market-clearing-price/"
DEFAULT_MD_FILE_PREFIX = "/var/www/html/wp-content/uploads/downloads/data/MPRESERVE/MP_RESERVE_"
DEFAULT_MD_FILE_SUFFIX = ".csv"
RAW_HEADER = [
    "RUN_TIME",
    "MKT_TYPE",
    "TIME_INTERVAL",
    "REGION_NAME",
    "RESOURCE_NAME",
    "RESOURCE_TYPE",
    "COMMODITY_TYPE",
    "MARGINAL_PRICE",
    "",
]
NORMALIZED_HEADER = RAW_HEADER[:-1]
TIMESTAMP_FORMATS = ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y")
EXPECTED_INTERVALS_PER_DAY = 288
USER_AGENT = "phl-market-data/0.1 (+https://www.iemop.ph/)"
LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class UrlPattern:
    page_url: str
    md_file_prefix: str
    md_file_suffix: str


@dataclass
class FileCheck:
    file_date: str
    status: str
    url: str
    local_path: str
    downloaded: bool
    http_status: int
    attachment_name: str
    bytes_downloaded: int
    row_count: int
    unique_intervals: int
    missing_intervals: int
    missing_intervals_sample: str
    duplicate_rows: int
    header_ok: bool
    eof_ok: bool
    mkt_types: str
    min_interval: str
    max_interval: str
    warnings: str
    error: str


def log(message: str) -> None:
    with LOG_LOCK:
        print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download daily RTD reserve market clearing price CSVs from IEMOP, "
            "validate them, and concatenate the range into one combined CSV."
        )
    )
    parser.add_argument("--start-url", help="First IEMOP download URL in the range.")
    parser.add_argument("--end-url", help="Last IEMOP download URL in the range.")
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD.")
    parser.add_argument(
        "--output-root",
        default="data/mp_reserve",
        help="Directory for raw downloads, manifests, and combined output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel download workers. Defaults to 6.",
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
        help="Re-download files even when a local raw CSV already exists.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Still write the combined CSV when some files fail QC.",
    )
    parser.add_argument(
        "--include-errors-in-combined",
        action="store_true",
        help="Include QC-error files in the combined CSV instead of only validated/warning files.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    using_urls = bool(args.start_url or args.end_url)
    using_dates = bool(args.start_date or args.end_date)

    if using_urls == using_dates:
        raise SystemExit(
            "Provide either --start-url/--end-url or --start-date/--end-date."
        )
    if using_urls and not (args.start_url and args.end_url):
        raise SystemExit("Both --start-url and --end-url are required together.")
    if using_dates and not (args.start_date and args.end_date):
        raise SystemExit("Both --start-date and --end-date are required together.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.timeout < 1:
        raise SystemExit("--timeout must be at least 1.")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1.")


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def decode_md_file_param(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding).decode("utf-8")


def encode_md_file_param(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def extract_file_date_from_path(md_file_path: str) -> date:
    filename = Path(md_file_path).name
    stem = Path(filename).stem
    try:
        return datetime.strptime(stem.rsplit("_", 1)[1], "%Y%m%d").date()
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not extract YYYYMMDD from {filename!r}.") from exc


def path_parts_from_decoded_path(md_file_path: str) -> tuple[str, str]:
    filename = Path(md_file_path).name
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    date_token = stem.rsplit("_", 1)[1]
    if len(date_token) != 8 or not date_token.isdigit():
        raise ValueError(f"Decoded path does not end with a YYYYMMDD filename: {md_file_path}")
    prefix = md_file_path[: -len(date_token + suffix)]
    return prefix, suffix


def parse_url_pattern_and_range(start_url: str, end_url: str) -> tuple[UrlPattern, date, date]:
    start_parsed = urllib.parse.urlparse(start_url)
    end_parsed = urllib.parse.urlparse(end_url)
    start_page = urllib.parse.urlunparse(
        (start_parsed.scheme, start_parsed.netloc, start_parsed.path, "", "", "")
    )
    end_page = urllib.parse.urlunparse(
        (end_parsed.scheme, end_parsed.netloc, end_parsed.path, "", "", "")
    )
    if start_page != end_page:
        raise ValueError("Start and end URLs do not point to the same download page.")

    start_query = urllib.parse.parse_qs(start_parsed.query)
    end_query = urllib.parse.parse_qs(end_parsed.query)
    try:
        start_md_file = decode_md_file_param(start_query["md_file"][0])
        end_md_file = decode_md_file_param(end_query["md_file"][0])
    except (KeyError, IndexError) as exc:
        raise ValueError("Each URL must include exactly one md_file query parameter.") from exc

    start_prefix, start_suffix = path_parts_from_decoded_path(start_md_file)
    end_prefix, end_suffix = path_parts_from_decoded_path(end_md_file)
    if start_prefix != end_prefix or start_suffix != end_suffix:
        raise ValueError("Start and end URLs do not share the same underlying file pattern.")

    start_date = extract_file_date_from_path(start_md_file)
    end_date = extract_file_date_from_path(end_md_file)
    return UrlPattern(start_page, start_prefix, start_suffix), start_date, end_date


def build_url(pattern: UrlPattern, current_date: date) -> str:
    md_file_path = (
        f"{pattern.md_file_prefix}{current_date.strftime('%Y%m%d')}{pattern.md_file_suffix}"
    )
    md_file = encode_md_file_param(md_file_path)
    return f"{pattern.page_url}?{urllib.parse.urlencode({'md_file': md_file})}"


def daterange(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def parse_market_timestamp(value: str) -> datetime:
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {value!r}")


def expected_intervals_for_day(file_date: date) -> list[datetime]:
    start_interval = datetime.combine(file_date, dt_time(hour=0, minute=5))
    return [start_interval + timedelta(minutes=5 * idx) for idx in range(EXPECTED_INTERVALS_PER_DAY)]


def request_download(url: str, timeout: int) -> tuple[bytes, int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        status = getattr(response, "status", 200)
        attachment_name = response.headers.get_filename() or ""
        return payload, status, attachment_name


def download_to_path(
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


def validate_csv(path: Path, file_date: date, url: str, attachment_name: str, http_status: int, downloaded: bool) -> FileCheck:
    warnings: list[str] = []
    local_path = str(path)
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return build_error_result(
                    file_date=file_date,
                    url=url,
                    path=local_path,
                    downloaded=downloaded,
                    http_status=http_status,
                    attachment_name=attachment_name,
                    bytes_downloaded=path.stat().st_size if path.exists() else 0,
                    error="CSV is empty.",
                )

            if header == NORMALIZED_HEADER:
                warnings.append("header_missing_trailing_blank_column")
                header_ok = True
            elif header == RAW_HEADER:
                header_ok = True
            else:
                return build_error_result(
                    file_date=file_date,
                    url=url,
                    path=local_path,
                    downloaded=downloaded,
                    http_status=http_status,
                    attachment_name=attachment_name,
                    bytes_downloaded=path.stat().st_size,
                    error=f"Unexpected header: {header!r}",
                )

            row_count = 0
            duplicate_rows = 0
            eof_ok = False
            mkt_types: set[str] = set()
            unique_intervals: set[datetime] = set()
            min_interval: Optional[datetime] = None
            max_interval: Optional[datetime] = None
            seen_rows: set[tuple[str, ...]] = set()

            for raw_row in reader:
                if not raw_row:
                    continue
                if raw_row == ["EOF"]:
                    eof_ok = True
                    break

                if len(raw_row) == len(NORMALIZED_HEADER):
                    normalized_row = raw_row + [""]
                    warnings.append("data_rows_missing_trailing_blank_column")
                elif len(raw_row) == len(RAW_HEADER):
                    normalized_row = raw_row
                else:
                    return build_error_result(
                        file_date=file_date,
                        url=url,
                        path=local_path,
                        downloaded=downloaded,
                        http_status=http_status,
                        attachment_name=attachment_name,
                        bytes_downloaded=path.stat().st_size,
                        error=f"Unexpected row length {len(raw_row)}: {raw_row!r}",
                    )

                if normalized_row[-1] not in ("", None):
                    return build_error_result(
                        file_date=file_date,
                        url=url,
                        path=local_path,
                        downloaded=downloaded,
                        http_status=http_status,
                        attachment_name=attachment_name,
                        bytes_downloaded=path.stat().st_size,
                        error=f"Expected an empty trailing column, got {normalized_row[-1]!r}.",
                    )

                try:
                    run_time = parse_market_timestamp(normalized_row[0])
                    time_interval = parse_market_timestamp(normalized_row[2])
                    float(normalized_row[7])
                except ValueError as exc:
                    return build_error_result(
                        file_date=file_date,
                        url=url,
                        path=local_path,
                        downloaded=downloaded,
                        http_status=http_status,
                        attachment_name=attachment_name,
                        bytes_downloaded=path.stat().st_size,
                        error=str(exc),
                    )

                if run_time.date() != file_date:
                    return build_error_result(
                        file_date=file_date,
                        url=url,
                        path=local_path,
                        downloaded=downloaded,
                        http_status=http_status,
                        attachment_name=attachment_name,
                        bytes_downloaded=path.stat().st_size,
                        error=f"RUN_TIME date {run_time.date()} does not match file date {file_date}.",
                    )

                key = tuple(normalized_row[: len(NORMALIZED_HEADER)])
                if key in seen_rows:
                    duplicate_rows += 1
                else:
                    seen_rows.add(key)

                mkt_types.add(normalized_row[1])
                unique_intervals.add(time_interval)
                min_interval = time_interval if min_interval is None else min(min_interval, time_interval)
                max_interval = time_interval if max_interval is None else max(max_interval, time_interval)
                row_count += 1

            if not eof_ok:
                warnings.append("missing_eof_marker")

            if not row_count:
                return build_error_result(
                    file_date=file_date,
                    url=url,
                    path=local_path,
                    downloaded=downloaded,
                    http_status=http_status,
                    attachment_name=attachment_name,
                    bytes_downloaded=path.stat().st_size,
                    error="CSV contained a header but no data rows.",
                )

            if mkt_types != {"RTD"}:
                return build_error_result(
                    file_date=file_date,
                    url=url,
                    path=local_path,
                    downloaded=downloaded,
                    http_status=http_status,
                    attachment_name=attachment_name,
                    bytes_downloaded=path.stat().st_size,
                    error=f"Unexpected MKT_TYPE values: {sorted(mkt_types)!r}",
                )

            if attachment_name and attachment_name != path.name:
                warnings.append(f"attachment_name_mismatch:{attachment_name}")

            expected_intervals = expected_intervals_for_day(file_date)
            missing_intervals = [dt for dt in expected_intervals if dt not in unique_intervals]
            if missing_intervals:
                warnings.append(f"missing_intervals:{len(missing_intervals)}")
            if duplicate_rows:
                warnings.append(f"duplicate_rows:{duplicate_rows}")

            status = "ok" if not warnings else "warning"
            return FileCheck(
                file_date=file_date.isoformat(),
                status=status,
                url=url,
                local_path=local_path,
                downloaded=downloaded,
                http_status=http_status,
                attachment_name=attachment_name or path.name,
                bytes_downloaded=path.stat().st_size,
                row_count=row_count,
                unique_intervals=len(unique_intervals),
                missing_intervals=len(missing_intervals),
                missing_intervals_sample=";".join(
                    dt.strftime("%Y-%m-%d %H:%M") for dt in missing_intervals[:10]
                ),
                duplicate_rows=duplicate_rows,
                header_ok=header_ok,
                eof_ok=eof_ok,
                mkt_types="|".join(sorted(mkt_types)),
                min_interval=min_interval.isoformat(sep=" ") if min_interval else "",
                max_interval=max_interval.isoformat(sep=" ") if max_interval else "",
                warnings="|".join(dict.fromkeys(warnings)),
                error="",
            )
    except FileNotFoundError as exc:
        return build_error_result(
            file_date=file_date,
            url=url,
            path=local_path,
            downloaded=downloaded,
            http_status=http_status,
            attachment_name=attachment_name,
            bytes_downloaded=0,
            error=str(exc),
        )


def build_error_result(
    *,
    file_date: date,
    url: str,
    path: str,
    downloaded: bool,
    http_status: int,
    attachment_name: str,
    bytes_downloaded: int,
    error: str,
) -> FileCheck:
    return FileCheck(
        file_date=file_date.isoformat(),
        status="error",
        url=url,
        local_path=path,
        downloaded=downloaded,
        http_status=http_status,
        attachment_name=attachment_name,
        bytes_downloaded=bytes_downloaded,
        row_count=0,
        unique_intervals=0,
        missing_intervals=0,
        missing_intervals_sample="",
        duplicate_rows=0,
        header_ok=False,
        eof_ok=False,
        mkt_types="",
        min_interval="",
        max_interval="",
        warnings="",
        error=error,
    )


def download_and_check(
    file_date: date,
    url: str,
    raw_dir: Path,
    timeout: int,
    retries: int,
    force: bool,
) -> FileCheck:
    destination = raw_dir / f"MP_RESERVE_{file_date.strftime('%Y%m%d')}.csv"
    file_label = file_date.isoformat()
    downloaded = False
    http_status = 0
    attachment_name = ""
    bytes_downloaded = destination.stat().st_size if destination.exists() else 0
    log(f"[{file_label}] download stage starting")
    try:
        downloaded, http_status, attachment_name, bytes_downloaded = download_to_path(
            url=url,
            destination=destination,
            timeout=timeout,
            retries=retries,
            force=force,
        )
        if downloaded:
            log(
                f"[{file_label}] download complete http={http_status} bytes={bytes_downloaded}; starting QC"
            )
        else:
            log(f"[{file_label}] using existing raw file; starting QC")
    except urllib.error.HTTPError as exc:
        result = build_error_result(
            file_date=file_date,
            url=url,
            path=str(destination),
            downloaded=downloaded,
            http_status=exc.code,
            attachment_name=attachment_name,
            bytes_downloaded=destination.stat().st_size if destination.exists() else 0,
            error=f"HTTP {exc.code}: {exc.reason}",
        )
        log(f"[{file_label}] error http={result.http_status} detail={result.error}")
        return result
    except Exception as exc:  # noqa: BLE001
        result = build_error_result(
            file_date=file_date,
            url=url,
            path=str(destination),
            downloaded=downloaded,
            http_status=http_status,
            attachment_name=attachment_name,
            bytes_downloaded=destination.stat().st_size if destination.exists() else 0,
            error=str(exc),
        )
        log(f"[{file_label}] error detail={result.error}")
        return result

    result = validate_csv(
        path=destination,
        file_date=file_date,
        url=url,
        attachment_name=attachment_name,
        http_status=http_status,
        downloaded=downloaded,
    )
    if result.status == "error":
        log(f"[{file_label}] QC failed detail={result.error}")
    else:
        warning_suffix = f" warnings={result.warnings}" if result.warnings else ""
        log(
            f"[{file_label}] QC {result.status} rows={result.row_count} "
            f"intervals={result.unique_intervals} missing={result.missing_intervals}{warning_suffix}"
        )
    return result


def write_manifest(results: list[FileCheck], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else list(asdict(FileCheck(
        file_date="",
        status="",
        url="",
        local_path="",
        downloaded=False,
        http_status=0,
        attachment_name="",
        bytes_downloaded=0,
        row_count=0,
        unique_intervals=0,
        missing_intervals=0,
        missing_intervals_sample="",
        duplicate_rows=0,
        header_ok=False,
        eof_ok=False,
        mkt_types="",
        min_interval="",
        max_interval="",
        warnings="",
        error="",
    )).keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_combined_parquet(
    results: list[FileCheck],
    combined_path: Path,
    include_error_files: bool = False,
) -> None:
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    valid_files = [
        Path(result.local_path)
        for result in results
        if include_error_files or result.status != "error"
    ]
    for csv_path in valid_files:
        frame = pd.read_csv(
            csv_path,
            usecols=list(range(len(NORMALIZED_HEADER))),
            dtype={
                "MKT_TYPE": "string",
                "REGION_NAME": "string",
                "RESOURCE_NAME": "string",
                "RESOURCE_TYPE": "string",
                "COMMODITY_TYPE": "string",
            },
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["RUN_TIME"] != "EOF"].copy()
    combined["RUN_TIME"] = pd.to_datetime(combined["RUN_TIME"], format="mixed")
    combined["TIME_INTERVAL"] = pd.to_datetime(combined["TIME_INTERVAL"], format="mixed")
    for column in [
        "MKT_TYPE",
        "REGION_NAME",
        "RESOURCE_NAME",
        "RESOURCE_TYPE",
        "COMMODITY_TYPE",
    ]:
        combined[column] = combined[column].astype("category")

    combined.to_parquet(combined_path, index=False, compression="zstd", engine="pyarrow")


def summarize(results: list[FileCheck]) -> str:
    ok_count = sum(result.status == "ok" for result in results)
    warning_count = sum(result.status == "warning" for result in results)
    error_count = sum(result.status == "error" for result in results)
    downloaded_count = sum(result.downloaded for result in results)
    total_rows = sum(result.row_count for result in results if result.status != "error")
    total_missing_intervals = sum(result.missing_intervals for result in results)
    return (
        f"files={len(results)} downloaded={downloaded_count} ok={ok_count} "
        f"warning={warning_count} error={error_count} rows={total_rows} "
        f"missing_intervals={total_missing_intervals}"
    )


def main() -> int:
    args = parse_args()

    if args.start_url:
        url_pattern, start_date, end_date = parse_url_pattern_and_range(args.start_url, args.end_url)
    else:
        start_date = parse_iso_date(args.start_date)
        end_date = parse_iso_date(args.end_date)
        url_pattern = UrlPattern(
            page_url=BASE_DOWNLOAD_PAGE,
            md_file_prefix=DEFAULT_MD_FILE_PREFIX,
            md_file_suffix=DEFAULT_MD_FILE_SUFFIX,
        )

    if start_date > end_date:
        raise SystemExit("Start date must be on or before end date.")

    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    qc_dir = output_root / "qc"
    combined_dir = output_root / "combined"
    results: list[FileCheck] = []

    work_items = [(current_date, build_url(url_pattern, current_date)) for current_date in daterange(start_date, end_date)]
    log(
        f"Preparing full run for {len(work_items)} files from {start_date.isoformat()} "
        f"through {end_date.isoformat()} into {output_root}"
    )
    log("Stage 1/3: download + per-file QC")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                download_and_check,
                file_date=current_date,
                url=url,
                raw_dir=raw_dir,
                timeout=args.timeout,
                retries=args.retries,
                force=args.force,
            ): current_date
            for current_date, url in work_items
        }
        for completed_count, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            results.append(result)
            log(
                f"Progress {completed_count}/{len(work_items)}: "
                f"{result.file_date} finished with status={result.status}"
            )

    results.sort(key=lambda item: item.file_date)

    manifest_name = f"mp_reserve_qc_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    manifest_path = qc_dir / manifest_name
    log("Stage 2/3: writing QC manifest")
    write_manifest(results, manifest_path)

    had_errors = any(result.status == "error" for result in results)
    combined_name = f"MP_RESERVE_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
    combined_path = combined_dir / combined_name

    if had_errors and not args.allow_partial:
        log(summarize(results))
        log(f"QC manifest: {manifest_path}")
        log("Stage 3/3: skipped combined parquet because at least one file failed")
        log("Re-run with --allow-partial to override.")
        return 1

    log("Stage 3/3: writing combined parquet")
    write_combined_parquet(
        results,
        combined_path,
        include_error_files=args.include_errors_in_combined,
    )
    log(summarize(results))
    log(f"QC manifest: {manifest_path}")
    log(f"Combined parquet: {combined_path}")
    return 0 if not had_errors else 1


if __name__ == "__main__":
    sys.exit(main())
