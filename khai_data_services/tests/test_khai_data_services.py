# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import khai_data_services
from khai.core.cv import container


def test_import():
    assert khai_data_services.__name__ == "khai_data_services"
    assert container.CONTRACT_VERSION >= 1
