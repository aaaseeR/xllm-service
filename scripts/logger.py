# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm-service/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Process-wide glog-style Python logger shared by Service utilities."""

from __future__ import annotations

from datetime import datetime
import logging
import os


class _GlogStyleFormatter(logging.Formatter):
    _LEVEL_MAP = {
        logging.DEBUG: "D",
        logging.INFO: "I",
        logging.WARNING: "W",
        logging.ERROR: "E",
        logging.FATAL: "F",
    }

    def format(self, record: logging.LogRecord) -> str:
        level = self._LEVEL_MAP.get(record.levelno, "I")
        now = datetime.fromtimestamp(record.created)
        timestamp = now.strftime("%Y%m%d %H:%M:%S")
        prefix = (
            f"{level}{timestamp}.{now.microsecond:06d} "
            f"{os.getpid()} {record.filename}:{record.lineno}]"
        )
        output = f"{prefix} {record.getMessage()}"
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            output += ("" if output.endswith("\n") else "\n") + record.exc_text
        if record.stack_info:
            output += ("" if output.endswith("\n") else "\n") + self.formatStack(
                record.stack_info
            )
        return output


logger = logging.getLogger("xllm")
logger.propagate = False

if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
    _handler = logging.StreamHandler()
    _handler.setFormatter(_GlogStyleFormatter())
    logger.addHandler(_handler)


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(level)


configure_logging("INFO")
