#!/usr/bin/env python3
from __future__ import annotations

import sys

from rtd_dataset_configs import RTDHS_CONFIG
from rtd_download_core import run_pipeline


if __name__ == "__main__":
    sys.exit(run_pipeline(RTDHS_CONFIG))
