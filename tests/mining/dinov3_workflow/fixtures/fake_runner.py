#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic external four-verb runner used by contract tests."""

from __future__ import annotations

import json
import sys


def main() -> int:
    """Return one normalized platform response."""
    mode, verb = sys.argv[1:3]
    client_job_id = (
        sys.argv[sys.argv.index("--client-job-id") + 1]
        if "--client-job-id" in sys.argv else None
    )
    if verb == "submit":
        result = {"state": "PENDING"}
    elif verb == "status":
        result = {
            "client_job_id": client_job_id,
            "state": "UNKNOWN" if mode == "unknown" else "COMPLETE",
            "backend_ref": "fake:1",
            "return_code": 0,
            "native_state": "SUCCEEDED",
        }
    elif verb == "logs":
        result = {"text": "external log", "cursor": "12"}
    else:
        result = {"state": "CANCELED", "client_job_id": client_job_id}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
