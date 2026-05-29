# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define entrypoint to run tasks for TMM mining."""

import argparse

from nvidia_tao_ds.mining.tmm import scripts
from nvidia_tao_ds.core.entrypoint.entrypoint import get_subtasks, launch, command_line_parser


def get_subtask_list():
    """Return the list of subtasks by inspecting the scripts package."""
    return get_subtasks(scripts)


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "tmm",
        add_help=True,
        description="TMM mining entrypoint",
    )

    # Build list of subtasks by inspecting the scripts package.
    subtasks = get_subtask_list()

    args, unknown_args = command_line_parser(parser, subtasks)

    # Parse the arguments and launch the subtask.
    launch(vars(args), unknown_args, subtasks, network="tmm")


if __name__ == "__main__":
    main()
