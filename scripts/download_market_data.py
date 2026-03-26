#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import csv
import re
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from rtd_dataset_configs import DATASET_CONFIGS
from rtd_download_core import (
    download_to_path,
    encode_md_file_param,
    log,
    normalize_data_row,
    parse_timestamp,
    run_pipeline,
)

REGION_CODES = ("CLUZ", "CVIS", "CMIN")
RTD_DATASET_CODE = "RTD"
RTD_PAGE_URL = "https://www.iemop.ph/market-data/rtd-prices-and-schedules/"
RTD_MD_FILE_PREFIX = "/var/www/html/wp-content/uploads/downloads/data/RTD/RTD_"
RTD_MD_FILE_SUFFIX = ".zip"
RTD_RAW_HEADER = (
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
RTD_NORMALIZED_HEADER = RTD_RAW_HEADER[:-1]
RTD_NUMERIC_COLUMNS = {"SCHED_MW", "LMP", "LOSS_FACTOR", "LMP_SMP", "LMP_LOSS", "LMP_CONGESTION"}
DEFAULT_DATASETS = ("RTDREG", "MP", "RTD", "RTDCV", "RTDHS")
TIMESTAMP_TOKEN_RE = re.compile(r"(\d{8,12})")


@dataclass(frozen=True)
class HourlyUrlPattern:
    page_url: str
    md_file_prefix: str
    md_file_suffix: str


@dataclass
class RtdFileCheck:
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
    missing_region_groups: int
    nonunique_lmp_smp_groups: int
    region_names: str
    min_interval: str
    max_interval: str
    warnings: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download RTDREG, MP, RTD, RTDCV, and RTDHS for a trailing window ending on --end-date, "
            "write cleaned combined parquet outputs, and run MP-vs-RTD LMP_SMP verification."
        )
    )
    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
        help="Window end date in YYYY-MM-DD. Defaults to today.",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=3,
        help="Trailing calendar months to include, anchored on --end-date. Defaults to 3.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DEFAULT_DATASETS),
        default=list(DEFAULT_DATASETS),
        help="Subset of retained datasets to download. Defaults to all retained datasets.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel workers for daily and hourly downloads. Defaults to 6.",
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
        help="Re-download files even when raw downloads already exist.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write combined outputs even when some source files fail QC.",
    )
    parser.add_argument(
        "--include-errors-in-combined",
        action="store_true",
        help="Include QC-error daily files in combined parquet outputs.",
    )
    args = parser.parse_args()
    if args.lookback_months < 0:
        raise SystemExit("--lookback-months must be at least 0.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.timeout < 1:
        raise SystemExit("--timeout must be at least 1.")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1.")
    return args


def subtract_months(anchor: date, months: int) -> date:
    month_index = (anchor.year * 12 + anchor.month - 1) - months
    year = month_index // 12
    month = month_index % 12 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def build_daily_args(
    *,
    start_date: date,
    end_date: date,
    output_root: str,
    workers: int,
    timeout: int,
    retries: int,
    force: bool,
    allow_partial: bool,
    include_errors_in_combined: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        start_url=None,
        end_url=None,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        output_root=output_root,
        workers=workers,
        timeout=timeout,
        retries=retries,
        force=force,
        allow_partial=allow_partial,
        include_errors_in_combined=include_errors_in_combined,
    )


def latest_matching_file(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} under {root}.")

    def sort_key(path: Path) -> tuple[str, str, str]:
        tokens = TIMESTAMP_TOKEN_RE.findall(path.stem)
        if not tokens:
            return ("", "", path.name)
        if len(tokens) == 1:
            return (tokens[0], tokens[0], path.name)
        return (tokens[-1], tokens[0], path.name)

    return max(matches, key=sort_key)


def build_rtd_url(pattern: HourlyUrlPattern, current_dt: datetime) -> str:
    md_file_path = f"{pattern.md_file_prefix}{current_dt.strftime('%Y%m%d%H%M')}{pattern.md_file_suffix}"
    md_file = encode_md_file_param(md_file_path)
    return f"{pattern.page_url}?{urllib.parse.urlencode({'md_file': md_file})}"


def extract_single_csv(zip_path: Path, csv_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        members = [member for member in archive.namelist() if not member.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one file inside {zip_path.name}, found {len(members)}.")
        member_name = members[0]
        if Path(member_name).suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV inside {zip_path.name}, found {member_name!r}.")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member_name) as source, csv_path.open("wb") as destination:
            destination.write(source.read())
        return member_name


def _rtd_config_proxy() -> object:
    class ConfigProxy:
        raw_header = RTD_RAW_HEADER
        normalized_header = RTD_NORMALIZED_HEADER
        raw_column_count = len(RTD_RAW_HEADER)
        normalized_column_count = len(RTD_NORMALIZED_HEADER)

    return ConfigProxy()


def validate_and_aggregate_rtd(
    csv_path: Path,
    attachment_name: str,
    zip_member_name: str,
) -> tuple[list[dict[str, object]], int, int, int, int, int, str, str, str, str]:
    warnings: list[str] = []
    raw_row_count = 0
    data_row_count = 0
    interval_values: set[datetime] = set()
    region_names: set[str] = set()
    lmp_smp_values: dict[tuple[datetime, str], set[float]] = {}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
            raw_row_count += 1
        except StopIteration as exc:
            raise ValueError("CSV is empty.") from exc

        if header == list(RTD_NORMALIZED_HEADER):
            warnings.append("header_missing_trailing_blank_column")
        elif header != list(RTD_RAW_HEADER):
            raise ValueError(f"Unexpected header: {header!r}")

        eof_ok = False
        for raw_row in reader:
            raw_row_count += 1
            if not raw_row:
                continue
            if raw_row == ["EOF"]:
                eof_ok = True
                break

            normalized_row = normalize_data_row(raw_row, _rtd_config_proxy(), warnings)
            row_map = dict(zip(RTD_NORMALIZED_HEADER, normalized_row))

            if str(row_map["MKT_TYPE"]) != "RTD":
                raise ValueError(f"Unexpected MKT_TYPE value {row_map['MKT_TYPE']!r}.")
            region_name = str(row_map["REGION_NAME"])
            if region_name not in REGION_CODES:
                continue

            try:
                interval_value = parse_timestamp(str(row_map["TIME_INTERVAL"]))
            except ValueError as exc:
                raise ValueError(f"Unsupported TIME_INTERVAL {row_map['TIME_INTERVAL']!r}") from exc

            for column in RTD_NUMERIC_COLUMNS:
                value = str(row_map[column])
                if value == "":
                    raise ValueError(f"Blank numeric value in {column}.")
                try:
                    row_map[column] = float(value)
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric value for {column}: {value!r}") from exc

            key = (interval_value, region_name)
            lmp_smp_values.setdefault(key, set()).add(float(row_map["LMP_SMP"]))
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
        return ([], raw_row_count, data_row_count, 0, 0, 0, "", "", "", "|".join(dict.fromkeys(warnings)))

    nonunique_groups = {key: values for key, values in lmp_smp_values.items() if len(values) != 1}
    if nonunique_groups:
        sample = ", ".join(
            f"{interval.isoformat(sep=' ')}|{region}|{sorted(values)}"
            for (interval, region), values in list(nonunique_groups.items())[:5]
        )
        raise ValueError(
            f"Non-unique LMP_SMP values found for {len(nonunique_groups)} interval-region groups: {sample}"
        )

    missing_region_groups = 0
    for interval_value in interval_values:
        present_regions = {region for (interval, region) in lmp_smp_values if interval == interval_value}
        missing_region_groups += len(set(REGION_CODES) - present_regions)
    if missing_region_groups:
        warnings.append(f"missing_region_groups:{missing_region_groups}")

    output_rows = [
        {
            "TIME_INTERVAL": interval_value,
            "REGION_NAME": region_name,
            "LMP_SMP": next(iter(values)),
        }
        for (interval_value, region_name), values in sorted(lmp_smp_values.items())
    ]
    return (
        output_rows,
        raw_row_count,
        data_row_count,
        len(interval_values),
        missing_region_groups,
        0,
        "|".join(sorted(region_names)),
        min(interval_values).isoformat(sep=" "),
        max(interval_values).isoformat(sep=" "),
        "|".join(dict.fromkeys(warnings)),
    )


def process_rtd_hourly_file(
    *,
    current_dt: datetime,
    pattern: HourlyUrlPattern,
    raw_dir: Path,
    timeout: int,
    retries: int,
    force: bool,
) -> tuple[RtdFileCheck, list[dict[str, object]]]:
    token = current_dt.strftime("%Y%m%d%H%M")
    url = build_rtd_url(pattern, current_dt)
    zip_path = raw_dir / f"RTD_{token}.zip"
    csv_path = raw_dir / f"RTD_{token}.csv"
    downloaded = False
    http_status = 0
    attachment_name = ""
    bytes_downloaded = zip_path.stat().st_size if zip_path.exists() else 0

    try:
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
                    f"[{RTD_DATASET_CODE} {token}] download complete "
                    f"http={http_status} bytes={bytes_downloaded}; extracting CSV"
                )
            else:
                log(f"[{RTD_DATASET_CODE} {token}] using existing ZIP; extracting CSV")
            zip_member_name = extract_single_csv(zip_path, csv_path)

        (
            output_rows,
            raw_row_count,
            data_row_count,
            interval_count,
            missing_region_groups,
            nonunique_lmp_smp_groups,
            region_names,
            min_interval,
            max_interval,
            warnings,
        ) = validate_and_aggregate_rtd(csv_path, attachment_name, zip_member_name)

        status = "ok" if not warnings else "warning"
        result = RtdFileCheck(
            file_token=token,
            status=status,
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
            missing_region_groups=missing_region_groups,
            nonunique_lmp_smp_groups=nonunique_lmp_smp_groups,
            region_names=region_names,
            min_interval=min_interval,
            max_interval=max_interval,
            warnings=warnings,
            error="",
        )
        return result, output_rows
    except Exception as exc:  # noqa: BLE001
        result = RtdFileCheck(
            file_token=token,
            status="error",
            url=url,
            zip_path=str(zip_path),
            csv_path=str(csv_path),
            downloaded=downloaded,
            http_status=getattr(exc, "code", http_status),
            attachment_name=attachment_name,
            zip_member_name="",
            bytes_downloaded=bytes_downloaded,
            raw_row_count=0,
            data_row_count=0,
            output_row_count=0,
            interval_count=0,
            missing_region_groups=0,
            nonunique_lmp_smp_groups=0,
            region_names="",
            min_interval="",
            max_interval="",
            warnings="",
            error=str(exc),
        )
        return result, []


def write_rtd_manifest(results: list[RtdFileCheck], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else list(asdict(RtdFileCheck(
        file_token="",
        status="",
        url="",
        zip_path="",
        csv_path="",
        downloaded=False,
        http_status=0,
        attachment_name="",
        zip_member_name="",
        bytes_downloaded=0,
        raw_row_count=0,
        data_row_count=0,
        output_row_count=0,
        interval_count=0,
        missing_region_groups=0,
        nonunique_lmp_smp_groups=0,
        region_names="",
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


def write_rtd_combined(rows: list[dict[str, object]], parquet_path: Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            {
                "TIME_INTERVAL": pd.Series(dtype="datetime64[ns]"),
                "REGION_NAME": pd.Series(dtype="category"),
                "LMP_SMP": pd.Series(dtype="float64"),
            }
        )
    else:
        frame["TIME_INTERVAL"] = pd.to_datetime(frame["TIME_INTERVAL"], errors="raise")
        frame["REGION_NAME"] = frame["REGION_NAME"].astype("category")
        frame["LMP_SMP"] = pd.to_numeric(frame["LMP_SMP"], errors="raise").astype("float64")
        frame = frame.sort_values(["TIME_INTERVAL", "REGION_NAME"], kind="stable").reset_index(drop=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False, compression="zstd", engine="pyarrow")


def run_rtd_pipeline(args: argparse.Namespace, start_date: date, end_date: date) -> int:
    start_dt = datetime.combine(start_date, time(hour=0, minute=0))
    end_dt = datetime.combine(end_date, time(hour=23, minute=0))
    output_root = Path("data/rtd")
    raw_dir = output_root / "raw"
    qc_dir = output_root / "qc"
    combined_dir = output_root / "combined"
    pattern = HourlyUrlPattern(
        page_url=RTD_PAGE_URL,
        md_file_prefix=RTD_MD_FILE_PREFIX,
        md_file_suffix=RTD_MD_FILE_SUFFIX,
    )

    work_items: list[datetime] = []
    current_dt = start_dt
    while current_dt <= end_dt:
        work_items.append(current_dt)
        current_dt += timedelta(hours=1)

    results: list[RtdFileCheck] = []
    combined_rows: list[dict[str, object]] = []
    log(
        f"Preparing full run for {len(work_items)} {RTD_DATASET_CODE} files from "
        f"{start_dt.strftime('%Y-%m-%d %H:%M')} through {end_dt.strftime('%Y-%m-%d %H:%M')} into {output_root}"
    )
    log("Stage 1/3: download + per-file QC")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                process_rtd_hourly_file,
                current_dt=current_dt,
                pattern=pattern,
                raw_dir=raw_dir,
                timeout=args.timeout,
                retries=args.retries,
                force=args.force,
            ): current_dt
            for current_dt in work_items
        }
        for completed_count, future in enumerate(as_completed(future_map), start=1):
            result, output_rows = future.result()
            results.append(result)
            combined_rows.extend(output_rows)
            log(
                f"Progress {completed_count}/{len(work_items)}: "
                f"{result.file_token} finished with status={result.status}"
            )

    results.sort(key=lambda item: item.file_token)
    combined_rows.sort(key=lambda row: (row["TIME_INTERVAL"], row["REGION_NAME"]))

    start_token = start_dt.strftime("%Y%m%d%H%M")
    end_token = end_dt.strftime("%Y%m%d%H%M")
    manifest_path = qc_dir / f"rtd_lmp_smp_qc_{start_token}_{end_token}.csv"
    combined_path = combined_dir / f"RTD_LMP_SMP_{start_token}_{end_token}.parquet"

    log("Stage 2/3: writing QC manifest")
    write_rtd_manifest(results, manifest_path)

    had_errors = any(result.status == "error" for result in results)
    if had_errors and not args.allow_partial:
        log(f"QC manifest: {manifest_path}")
        log("Stage 3/3: skipped combined parquet because at least one file failed")
        log("Re-run with --allow-partial to override.")
        return 1

    log("Stage 3/3: writing combined parquet")
    write_rtd_combined(combined_rows, combined_path)
    log(f"QC manifest: {manifest_path}")
    log(f"Combined parquet: {combined_path}")
    return 0 if not had_errors else 1


def write_mp_rtd_verification(mp_path: Path, rtd_path: Path, output_path: Path) -> None:
    mp = pd.read_parquet(mp_path).copy()
    mp = mp.loc[
        (mp["COMMODITY_TYPE"] == "En") & mp["REGION_NAME"].isin(REGION_CODES),
        ["TIME_INTERVAL", "REGION_NAME", "MARGINAL_PRICE"],
    ].copy()
    mp["TIME_INTERVAL"] = pd.to_datetime(mp["TIME_INTERVAL"])

    rtd = pd.read_parquet(rtd_path).copy()
    rtd = rtd.loc[rtd["REGION_NAME"].isin(REGION_CODES), ["TIME_INTERVAL", "REGION_NAME", "LMP_SMP"]].copy()
    rtd["TIME_INTERVAL"] = pd.to_datetime(rtd["TIME_INTERVAL"])

    mp_unique_counts = mp.groupby(["TIME_INTERVAL", "REGION_NAME"], observed=True)["MARGINAL_PRICE"].nunique()
    mp_nonunique = mp_unique_counts[mp_unique_counts != 1]
    mp_unique = (
        mp.groupby(["TIME_INTERVAL", "REGION_NAME"], observed=True)["MARGINAL_PRICE"]
        .first()
        .rename("MARGINAL_PRICE")
        .reset_index()
    )
    if not mp_nonunique.empty:
        mp_unique = mp_unique.merge(
            mp_nonunique.rename("mp_unique_count").reset_index(),
            on=["TIME_INTERVAL", "REGION_NAME"],
            how="left",
        )
        mp_unique = mp_unique.loc[mp_unique["mp_unique_count"].isna()].drop(columns=["mp_unique_count"])

    rtd_unique_counts = rtd.groupby(["TIME_INTERVAL", "REGION_NAME"], observed=True)["LMP_SMP"].nunique()
    rtd_nonunique = rtd_unique_counts[rtd_unique_counts != 1]

    merged = mp_unique.merge(rtd, on=["TIME_INTERVAL", "REGION_NAME"], how="outer", indicator=True)
    overlap = merged.loc[merged["_merge"] == "both"].copy()
    overlap["abs_diff"] = (overlap["MARGINAL_PRICE"] - overlap["LMP_SMP"]).abs()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "status": "ok" if mp_nonunique.empty and rtd_nonunique.empty and int((overlap["abs_diff"] > 1e-9).sum()) == 0 else "warning",
                "mp_path": str(mp_path),
                "rtd_path": str(rtd_path),
                "overlap_groups": int(len(overlap)),
                "exact_match_groups": int((overlap["abs_diff"] <= 1e-9).sum()),
                "mismatch_groups": int((overlap["abs_diff"] > 1e-9).sum()),
                "max_abs_diff": float(overlap["abs_diff"].max()) if not overlap.empty else 0.0,
                "mp_only_groups": int((merged["_merge"] == "left_only").sum()),
                "rtd_only_groups": int((merged["_merge"] == "right_only").sum()),
                "mp_nonunique_groups": int(len(mp_nonunique)),
                "rtd_nonunique_groups": int(len(rtd_nonunique)),
                "mp_nonunique_sample": " | ".join(
                    f"{idx[0].strftime('%Y-%m-%d %H:%M:%S')}|{idx[1]}|{int(value)}"
                    for idx, value in mp_nonunique.head(5).items()
                ),
                "rtd_nonunique_sample": " | ".join(
                    f"{idx[0].strftime('%Y-%m-%d %H:%M:%S')}|{idx[1]}|{int(value)}"
                    for idx, value in rtd_nonunique.head(5).items()
                ),
                "mismatch_sample": " | ".join(
                    f"{row.TIME_INTERVAL.strftime('%Y-%m-%d %H:%M:%S')}|{row.REGION_NAME}|{row.MARGINAL_PRICE:.6f}|{row.LMP_SMP:.6f}"
                    for row in overlap.loc[overlap["abs_diff"] > 1e-9, ["TIME_INTERVAL", "REGION_NAME", "MARGINAL_PRICE", "LMP_SMP"]]
                    .head(5)
                    .itertuples(index=False)
                ),
            }
        ]
    )
    summary.to_csv(output_path, index=False)


def main() -> int:
    args = parse_args()
    end_date = date.fromisoformat(args.end_date)
    start_date = subtract_months(end_date, args.lookback_months)
    log(f"Window start={start_date.isoformat()} end={end_date.isoformat()} datasets={','.join(args.datasets)}")

    exit_codes: list[int] = []
    for dataset_code in args.datasets:
        if dataset_code == RTD_DATASET_CODE:
            exit_codes.append(run_rtd_pipeline(args, start_date, end_date))
            continue

        config = DATASET_CONFIGS[dataset_code]
        daily_args = build_daily_args(
            start_date=start_date,
            end_date=end_date,
            output_root=config.output_root,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
            force=args.force,
            allow_partial=args.allow_partial,
            include_errors_in_combined=args.include_errors_in_combined,
        )
        exit_codes.append(run_pipeline(config, daily_args))

    if {"MP", "RTD"}.issubset(set(args.datasets)):
        mp_path = latest_matching_file(Path("data/mp/combined"), "MP_*.parquet")
        rtd_path = latest_matching_file(Path("data/rtd/combined"), "RTD_LMP_SMP_*.parquet")
        verification_path = Path("data/mp/qc") / f"mp_vs_rtd_lmp_smp_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        write_mp_rtd_verification(mp_path, rtd_path, verification_path)
        log(f"MP-vs-RTD verification: {verification_path}")

    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
