# Copyright (c) 2026 Nokia Bell Labs
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os


def clear_logs(log_file):
    # Remove the log file if it exists to start fresh
    if os.path.exists(log_file):
        os.remove(log_file)


def setup_logging(config_dict):
    log_file = config_dict['log_file']
    clear_logs(log_file)

    if config_dict['logging_detail'] == "verbose":
        log_format = '%(asctime)s %(name)s %(levelname)s - %(message)s'
    elif config_dict['logging_detail'] == "simple":
        log_format = '%(message)s'
    else:
        raise ValueError(f"Invalid value given for config_dict['logging_detail']: {config_dict['logging_detail']}")

    logging.basicConfig(
        level=config_dict['logging_level'],
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler(log_file, mode='w')  # Output to a file
        ]
    )


def truncate_log_file(log_file):
    """Clear a log file without deleting it from the OS."""
    with open(log_file, "w") as f:
        pass
