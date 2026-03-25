#!/usr/bin/env python3
from __future__ import annotations

from rtd_download_core import DatasetConfig, IntervalExpectation

MP_CONFIG = DatasetConfig(
    dataset_code="MP",
    description="RTD market clearing price",
    page_url="https://www.iemop.ph/market-data/rtd-market-clearing-price/",
    md_file_prefix="/var/www/html/wp-content/uploads/downloads/data/MP/MP_",
    output_root="data/mp",
    raw_filename_prefix="MP",
    qc_manifest_prefix="mp_qc",
    combined_filename_prefix="MP",
    raw_header=(
        "RUN_TIME",
        "MKT_TYPE",
        "TIME_INTERVAL",
        "REGION_NAME",
        "RESOURCE_NAME",
        "RESOURCE_TYPE",
        "COMMODITY_TYPE",
        "MARGINAL_PRICE",
        "",
    ),
    timestamp_columns=("RUN_TIME", "TIME_INTERVAL"),
    numeric_columns=("MARGINAL_PRICE",),
    interval_expectation=IntervalExpectation(
        interval_column="TIME_INTERVAL",
        expected_count=288,
        required_values_by_column={
            "REGION_NAME": ("CLUZ", "CVIS", "CMIN"),
            "COMMODITY_TYPE": ("En",),
        },
    ),
)

RTDCV_CONFIG = DatasetConfig(
    dataset_code="RTDCV",
    description="RTD congestions manifesting",
    page_url="https://www.iemop.ph/market-data/congestions-manifesting-in-rtd/",
    md_file_prefix="/var/www/html/wp-content/uploads/downloads/data/RTDCV/RTDCV_",
    output_root="data/rtdcv",
    raw_filename_prefix="RTDCV",
    qc_manifest_prefix="rtdcv_qc",
    combined_filename_prefix="RTDCV",
    raw_header=(
        "RUN_TIME",
        "MKT_TYPE",
        "TIME_INTERVAL",
        "CONGEST_TYPE",
        "RUN_TYPE",
        "EQUIPMENT_NAME",
        "STATION_NAME",
        "VOLTAGE_LEVEL",
        "BINDING_LIMIT",
        "MW_FLOW",
        "OVERLOAD_MW",
        "PCT_MW",
        "",
    ),
    timestamp_columns=("RUN_TIME", "TIME_INTERVAL"),
    numeric_columns=("VOLTAGE_LEVEL", "BINDING_LIMIT", "MW_FLOW", "OVERLOAD_MW", "PCT_MW"),
)

RTDHS_CONFIG = DatasetConfig(
    dataset_code="RTDHS",
    description="RTD HVDC schedules",
    page_url="https://www.iemop.ph/market-data/rtd-hvdc-schedules/",
    md_file_prefix="/var/www/html/wp-content/uploads/downloads/data/RTDHS/RTDHS_",
    output_root="data/rtdhs",
    raw_filename_prefix="RTDHS",
    qc_manifest_prefix="rtdhs_qc",
    combined_filename_prefix="RTDHS",
    raw_header=(
        "RUN_TIME",
        "MKT_TYPE",
        "TIME_INTERVAL",
        "HVDC_NAME",
        "CONGESTION_FLAG",
        "FLOW_FROM",
        "FLOW_TO",
        "OVERLOAD_MW",
        "",
    ),
    timestamp_columns=("RUN_TIME", "TIME_INTERVAL"),
    numeric_columns=("FLOW_FROM", "FLOW_TO", "OVERLOAD_MW"),
    interval_expectation=IntervalExpectation(
        interval_column="TIME_INTERVAL",
        expected_count=288,
        required_values_by_column={"HVDC_NAME": ("MINVIS1", "VISLUZ1")},
    ),
)

RTDREG_CONFIG = DatasetConfig(
    dataset_code="RTDREG",
    description="RTD regional summaries",
    page_url="https://www.iemop.ph/market-data/rtd-regional-summaries/",
    md_file_prefix="/var/www/html/wp-content/uploads/downloads/data/RTDREG/RTDREG_",
    output_root="data/rtdreg",
    raw_filename_prefix="RTDREG",
    qc_manifest_prefix="rtdreg_qc",
    combined_filename_prefix="RTDREG",
    raw_header=(
        "RUN_TIME",
        "MKT_TYPE",
        "TIME_INTERVAL",
        "REGION_NAME",
        "COMMODITY_TYPE",
        "MKT_REQT",
        "LOAD_BID",
        "LOAD_CURTAILED",
        "LOSSES",
        "GENERATION",
        "MKT_IMPORT",
        "MKT_EXPORT",
        "",
    ),
    timestamp_columns=("RUN_TIME", "TIME_INTERVAL"),
    numeric_columns=(
        "MKT_REQT",
        "LOAD_BID",
        "LOAD_CURTAILED",
        "LOSSES",
        "GENERATION",
        "MKT_IMPORT",
        "MKT_EXPORT",
    ),
    interval_expectation=IntervalExpectation(
        interval_column="TIME_INTERVAL",
        expected_count=288,
        required_values_by_column={
            "REGION_NAME": ("CLUZ", "CVIS", "CMIN"),
            "COMMODITY_TYPE": ("En", "Dr", "Fr", "Rd", "Ru"),
        },
    ),
)

DATASET_CONFIGS = {
    config.dataset_code: config
    for config in (
        MP_CONFIG,
        RTDCV_CONFIG,
        RTDHS_CONFIG,
        RTDREG_CONFIG,
    )
}
