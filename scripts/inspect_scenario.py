"""Print traceable scenario definitions used by environment experiments."""

from __future__ import annotations

import argparse
import json

from env.config import DEFAULT_CONFIG
from env.scenario import (
    PROJECT_CORE_PROFILE,
    YOON_2023_CASES,
    YOON_2023_PROFILE,
    build_project_core_profile,
    build_yoon_2023_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=(PROJECT_CORE_PROFILE, YOON_2023_PROFILE),
        default=PROJECT_CORE_PROFILE,
    )
    parser.add_argument(
        "--case",
        choices=sorted(YOON_2023_CASES),
        default="case_4_1",
        help="Yoon 2023 experiment case; ignored for project_core",
    )
    args = parser.parse_args()

    if args.profile == YOON_2023_PROFILE:
        profile = build_yoon_2023_profile(args.case)
    else:
        profile = build_project_core_profile(DEFAULT_CONFIG)

    print(json.dumps(profile.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
