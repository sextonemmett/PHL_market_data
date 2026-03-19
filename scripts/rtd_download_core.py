#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

TIMESTAMP_FORMATS = (
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)
USER_AGENT = "phl-market-data/0.1 (+https://www.iemop.ph/)"
LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class UrlPattern:
    page_url: str
    md_file_prefix: str
    md_file_suffix: str


@dataclass(frozen=True)
class IntervalExpectation:
    interval_column: str
    expected_count: int
    required_values_by_column: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetConfig:
    dataset_code: str
    description: str
    page_url: str
    md_file_prefix: str
    output_root: str
    raw_filename_prefix: str
    qc_manifest_prefix: str
    combined_filename_prefix: str
    raw_header: tuple[str, ...]
    timestamp_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    md_file_suffix: str = ".csv"
    interval_expectation: Optional[IntervalExpectation] = None

    @property
    def normalized_header(self) -> tuple[str, ...]:
        return self.raw_header[:-1]

    @property
    def text_columns(self) -> tuple[str, ...]:
        excluded = set(self.timestamp_columns) | set(self.numeric_columns)
        return tuple(column for column in self.normalized_header if column not in excluded)

    @property
    def raw_column_count(self) -> int:
        return len(self.raw_header)

    @property
    def normalized_column_count(self) -> int:
        return len(self.normalized_header)


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
    raw_row_count: int
    data_row_count: int
    header_ok: bool
    eof_ok: bool
    empty_file: bool
    mkt_types: str
    min_timestamp: str
    max_timestamp: str
    warnings: str
    error: str


def log(message: str) -> None:
    with LOG_LOCK:
        print(message, flush=True)


def parse_args(config: DatasetConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Download daily {config.description} CSVs from IEMOP, validate them, "
            "and concatenate the range into one combined Parquet file."
        )
    )
    parser.add_argument("--start-url", help="First IEMOP download URL in the range.")
    parser.add_argument("--end-url", help="Last IEMOP download URL in the range.")
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD.")
    parser.add_argument(
        "--output-root",
        default=config.output_root,
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
        help="Still write the combined Parquet when some files fail QC.",
    )
    parser.add_argument(
        "--include-errors-in-combined",
        action="store_true",
        help="Include QC-error files in the combined Parquet instead of only validated/warning files.",
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
        raise ValueError(
            f"Decoded path does not end with a YYYYMMDD filename: {md_file_path}"
        )
    prefix = md_file_path[: -len(date_token + suffix)]
    return prefix, suffix


def parse_url_pattern_and_range(
    start_url: str,
    end_url: str,
) -> tuple[UrlPattern, date, date]:
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
        raise ValueError(
            "Each URL must include exactly one md_file query parameter."
        ) from exc

    start_prefix, start_suffix = path_parts_from_decoded_path(start_md_file)
    end_prefix, end_suffix = path_parts_from_decoded_path(end_md_file)
    if start_prefix != end_prefix or start_suffix != end_suffix:
        raise ValueError(
            "Start and end URLs do not share the same underlying file pattern."
        )

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


def parse_timestamp(value: str) -> datetime:
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {value!r}")


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


def empty_file_check() -> FileCheck:
    return FileCheck(
        file_date="",
        status="",
        url="",
        local_path="",
        downloaded=False,
        http_status=0,
        attachment_name="",
        bytes_downloaded=0,
        raw_row_count=0,
        data_row_count=0,
        header_ok=False,
        eof_ok=False,
        empty_file=False,
        mkt_types="",
        min_timestamp="",
        max_timestamp="",
        warnings="",
        error="",
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
    raw_row_count: int,
    header_ok: bool,
    eof_ok: bool,
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
        raw_row_count=raw_row_count,
        data_row_count=0,
        header_ok=header_ok,
        eof_ok=eof_ok,
        empty_file=False,
        mkt_types="",
        min_timestamp="",
        max_timestamp="",
        warnings="",
        error=error,
    )


def normalize_data_row(
    raw_row: list[str],
    config: DatasetConfig,
    warnings: list[str],
) -> list[str]:
    if len(raw_row) == config.normalized_column_count:
        warnings.append("data_rows_missing_trailing_blank_column")
        return raw_row
    if len(raw_row) == config.raw_column_count:
        if raw_row[-1] not in ("", None):
            raise ValueError(
                f"Expected an empty trailing column, got {raw_row[-1]!r}."
            )
        return raw_row[:-1]
    raise ValueError(f"Unexpected row length {len(raw_row)}: {raw_row!r}")


def validate_interval_expectation(
    parsed_rows: list[dict[str, object]],
    config: DatasetConfig,
    warnings: list[str],
) -> None:
    expectation = config.interval_expectation
    if expectation is None or not parsed_rows:
        return

    unique_intervals = {
        row[expectation.interval_column]
        for row in parsed_rows
        if row.get(expectation.interval_column) is not None
    }
    if len(unique_intervals) != expectation.expected_count:
        warnings.append(
            f"unexpected_{expectation.interval_column.lower()}_count:{len(unique_intervals)}"
        )

    for column, expected_values in expectation.required_values_by_column.items():
        missing_interval_count = 0
        for interval in unique_intervals:
            seen_values = {
                str(row[column])
                for row in parsed_rows
                if row.get(expectation.interval_column) == interval and row.get(column) is not None
            }
            if any(value not in seen_values for value in expected_values):
                missing_interval_count += 1
        if missing_interval_count:
            warnings.append(
                f"missing_{column.lower()}_values:{missing_interval_count}"
            )


def validate_csv(
    path: Path,
    file_date: date,
    url: str,
    attachment_name: str,
    http_status: int,
    downloaded: bool,
    config: DatasetConfig,
) -> FileCheck:
    warnings: list[str] = []
    local_path = str(path)
    raw_row_count = 0
    timestamp_columns = set(config.timestamp_columns)
    numeric_columns = set(config.numeric_columns)
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
                raw_row_count += 1
            except StopIteration:
                return build_error_result(
                    file_date=file_date,
                    url=url,
                    path=local_path,
                    downloaded=downloaded,
                    http_status=http_status,
                    attachment_name=attachment_name,
                    bytes_downloaded=path.stat().st_size if path.exists() else 0,
                    raw_row_count=raw_row_count,
                    header_ok=False,
                    eof_ok=False,
                    error="CSV is empty.",
                )

            if header == list(config.normalized_header):
                warnings.append("header_missing_trailing_blank_column")
                header_ok = True
            elif header == list(config.raw_header):
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
                    raw_row_count=raw_row_count,
                    header_ok=False,
                    eof_ok=False,
                    error=f"Unexpected header: {header!r}",
                )

            data_row_count = 0
            eof_ok = False
            mkt_types: set[str] = set()
            min_timestamp: Optional[datetime] = None
            max_timestamp: Optional[datetime] = None
            parsed_rows: list[dict[str, object]] = []

            for raw_row in reader:
                raw_row_count += 1
                if not raw_row:
                    continue
                if raw_row == ["EOF"]:
                    eof_ok = True
                    break

                try:
                    normalized_row = normalize_data_row(raw_row, config, warnings)
                except ValueError as exc:
                    return build_error_result(
                        file_date=file_date,
                        url=url,
                        path=local_path,
                        downloaded=downloaded,
                        http_status=http_status,
                        attachment_name=attachment_name,
                        bytes_downloaded=path.stat().st_size,
                        raw_row_count=raw_row_count,
                        header_ok=header_ok,
                        eof_ok=eof_ok,
                        error=str(exc),
                    )

                row_map: dict[str, object] = {}
                for column, value in zip(config.normalized_header, normalized_row):
                    row_map[column] = value
                    if column in timestamp_columns:
                        try:
                            parsed_value = parse_timestamp(value)
                        except ValueError as exc:
                            return build_error_result(
                                file_date=file_date,
                                url=url,
                                path=local_path,
                                downloaded=downloaded,
                                http_status=http_status,
                                attachment_name=attachment_name,
                                bytes_downloaded=path.stat().st_size,
                                raw_row_count=raw_row_count,
                                header_ok=header_ok,
                                eof_ok=eof_ok,
                                error=str(exc),
                            )
                        row_map[column] = parsed_value
                        min_timestamp = (
                            parsed_value
                            if min_timestamp is None
                            else min(min_timestamp, parsed_value)
                        )
                        max_timestamp = (
                            parsed_value
                            if max_timestamp is None
                            else max(max_timestamp, parsed_value)
                        )
                    elif column in numeric_columns and value != "":
                        try:
                            float(value)
                        except ValueError as exc:
                            return build_error_result(
                                file_date=file_date,
                                url=url,
                                path=local_path,
                                downloaded=downloaded,
                                http_status=http_status,
                                attachment_name=attachment_name,
                                bytes_downloaded=path.stat().st_size,
                                raw_row_count=raw_row_count,
                                header_ok=header_ok,
                                eof_ok=eof_ok,
                                error=str(exc),
                            )

                mkt_type = str(row_map["MKT_TYPE"])
                mkt_types.add(mkt_type)
                parsed_rows.append(row_map)
                data_row_count += 1

            if not eof_ok:
                warnings.append("missing_eof_marker")

            if attachment_name and attachment_name != path.name:
                warnings.append(f"attachment_name_mismatch:{attachment_name}")

            if data_row_count == 0:
                warnings.append("empty_file")
            elif mkt_types != {"RTD"}:
                return build_error_result(
                    file_date=file_date,
                    url=url,
                    path=local_path,
                    downloaded=downloaded,
                    http_status=http_status,
                    attachment_name=attachment_name,
                    bytes_downloaded=path.stat().st_size,
                    raw_row_count=raw_row_count,
                    header_ok=header_ok,
                    eof_ok=eof_ok,
                    error=f"Unexpected MKT_TYPE values: {sorted(mkt_types)!r}",
                )

            validate_interval_expectation(parsed_rows, config, warnings)

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
                raw_row_count=raw_row_count,
                data_row_count=data_row_count,
                header_ok=header_ok,
                eof_ok=eof_ok,
                empty_file=data_row_count == 0,
                mkt_types="|".join(sorted(mkt_types)),
                min_timestamp=min_timestamp.isoformat(sep=" ") if min_timestamp else "",
                max_timestamp=max_timestamp.isoformat(sep=" ") if max_timestamp else "",
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
            raw_row_count=raw_row_count,
            header_ok=False,
            eof_ok=False,
            error=str(exc),
        )


def download_and_check(
    file_date: date,
    url: str,
    raw_dir: Path,
    timeout: int,
    retries: int,
    force: bool,
    config: DatasetConfig,
) -> FileCheck:
    destination = raw_dir / f"{config.raw_filename_prefix}_{file_date.strftime('%Y%m%d')}.csv"
    file_label = file_date.isoformat()
    downloaded = False
    http_status = 0
    attachment_name = ""
    bytes_downloaded = destination.stat().st_size if destination.exists() else 0
    log(f"[{config.dataset_code} {file_label}] download stage starting")
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
                f"[{config.dataset_code} {file_label}] download complete "
                f"http={http_status} bytes={bytes_downloaded}; starting QC"
            )
        else:
            log(f"[{config.dataset_code} {file_label}] using existing raw file; starting QC")
    except urllib.error.HTTPError as exc:
        result = build_error_result(
            file_date=file_date,
            url=url,
            path=str(destination),
            downloaded=downloaded,
            http_status=exc.code,
            attachment_name=attachment_name,
            bytes_downloaded=destination.stat().st_size if destination.exists() else 0,
            raw_row_count=0,
            header_ok=False,
            eof_ok=False,
            error=f"HTTP {exc.code}: {exc.reason}",
        )
        log(
            f"[{config.dataset_code} {file_label}] error http={result.http_status} "
            f"detail={result.error}"
        )
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
            raw_row_count=0,
            header_ok=False,
            eof_ok=False,
            error=str(exc),
        )
        log(f"[{config.dataset_code} {file_label}] error detail={result.error}")
        return result

    result = validate_csv(
        path=destination,
        file_date=file_date,
        url=url,
        attachment_name=attachment_name,
        http_status=http_status,
        downloaded=downloaded,
        config=config,
    )
    if result.status == "error":
        log(f"[{config.dataset_code} {file_label}] QC failed detail={result.error}")
    else:
        warning_suffix = f" warnings={result.warnings}" if result.warnings else ""
        log(
            f"[{config.dataset_code} {file_label}] QC {result.status} "
            f"data_rows={result.data_row_count} empty={result.empty_file}{warning_suffix}"
        )
    return result


def write_manifest(results: list[FileCheck], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else list(asdict(empty_file_check()).keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def load_normalized_frame(csv_path: Path, config: DatasetConfig) -> pd.DataFrame:
    rows: list[list[str]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame(columns=list(config.normalized_header))

        if header not in (list(config.raw_header), list(config.normalized_header)):
            raise ValueError(f"Unexpected header in {csv_path}: {header!r}")

        for raw_row in reader:
            if not raw_row or raw_row == ["EOF"]:
                if raw_row == ["EOF"]:
                    break
                continue
            rows.append(normalize_data_row(raw_row, config, []))

    return pd.DataFrame(rows, columns=list(config.normalized_header))


def build_empty_frame(config: DatasetConfig) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for column in config.normalized_header:
        if column in config.timestamp_columns:
            data[column] = pd.Series(dtype="datetime64[ns]")
        elif column in config.numeric_columns:
            data[column] = pd.Series(dtype="float64")
        else:
            data[column] = pd.Series(dtype="category")
    return pd.DataFrame(data, columns=list(config.normalized_header))


def apply_combined_dtypes(frame: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    if frame.empty:
        return build_empty_frame(config)

    combined = frame.copy()
    for column in config.timestamp_columns:
        combined[column] = pd.to_datetime(combined[column], format="mixed", errors="raise")
    for column in config.numeric_columns:
        combined[column] = pd.to_numeric(combined[column], errors="coerce").astype("float64")
    for column in config.text_columns:
        combined[column] = combined[column].astype("category")
    return combined[list(config.normalized_header)]


def write_combined_parquet(
    results: list[FileCheck],
    combined_path: Path,
    config: DatasetConfig,
    include_error_files: bool = False,
) -> None:
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    valid_files = [
        Path(result.local_path)
        for result in results
        if include_error_files or result.status != "error"
    ]

    if not valid_files:
        build_empty_frame(config).to_parquet(
            combined_path,
            index=False,
            compression="zstd",
            engine="pyarrow",
        )
        return

    frames = [load_normalized_frame(csv_path, config) for csv_path in valid_files]
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if non_empty_frames:
        combined = pd.concat(non_empty_frames, ignore_index=True)
        combined = apply_combined_dtypes(combined, config)
    else:
        combined = build_empty_frame(config)

    combined.to_parquet(combined_path, index=False, compression="zstd", engine="pyarrow")


def summarize(results: list[FileCheck]) -> str:
    ok_count = sum(result.status == "ok" for result in results)
    warning_count = sum(result.status == "warning" for result in results)
    error_count = sum(result.status == "error" for result in results)
    downloaded_count = sum(result.downloaded for result in results)
    total_rows = sum(result.data_row_count for result in results if result.status != "error")
    empty_count = sum(result.empty_file for result in results)
    return (
        f"files={len(results)} downloaded={downloaded_count} ok={ok_count} "
        f"warning={warning_count} error={error_count} data_rows={total_rows} "
        f"empty_files={empty_count}"
    )


def run_pipeline(config: DatasetConfig) -> int:
    args = parse_args(config)

    if args.start_url:
        url_pattern, start_date, end_date = parse_url_pattern_and_range(
            args.start_url,
            args.end_url,
        )
    else:
        start_date = parse_iso_date(args.start_date)
        end_date = parse_iso_date(args.end_date)
        url_pattern = UrlPattern(
            page_url=config.page_url,
            md_file_prefix=config.md_file_prefix,
            md_file_suffix=config.md_file_suffix,
        )

    if start_date > end_date:
        raise SystemExit("Start date must be on or before end date.")

    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    qc_dir = output_root / "qc"
    combined_dir = output_root / "combined"
    results: list[FileCheck] = []

    work_items = [
        (current_date, build_url(url_pattern, current_date))
        for current_date in daterange(start_date, end_date)
    ]
    log(
        f"Preparing full run for {len(work_items)} {config.dataset_code} files from "
        f"{start_date.isoformat()} through {end_date.isoformat()} into {output_root}"
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
                config=config,
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

    manifest_name = (
        f"{config.qc_manifest_prefix}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    )
    manifest_path = qc_dir / manifest_name
    log("Stage 2/3: writing QC manifest")
    write_manifest(results, manifest_path)

    had_errors = any(result.status == "error" for result in results)
    combined_name = (
        f"{config.combined_filename_prefix}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
    )
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
        config=config,
        include_error_files=args.include_errors_in_combined,
    )
    log(summarize(results))
    log(f"QC manifest: {manifest_path}")
    log(f"Combined parquet: {combined_path}")
    return 0 if not had_errors else 1
