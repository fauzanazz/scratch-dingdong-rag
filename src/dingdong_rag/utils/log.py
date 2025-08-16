from __future__ import annotations

import sys
from datetime import datetime
from typing import Any


class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


def _log(level: str, message: str, *args: Any) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = message if not args else message.format(*args)
    stream = sys.stderr if level in {LogLevel.WARNING, LogLevel.ERROR} else sys.stdout
    print(f"[{timestamp}] {level}: {text}", file=stream)


def debug(message: str, *args: Any) -> None:
    _log(LogLevel.DEBUG, message, *args)


def info(message: str, *args: Any) -> None:
    _log(LogLevel.INFO, message, *args)


def warning(message: str, *args: Any) -> None:
    _log(LogLevel.WARNING, message, *args)


def error(message: str, *args: Any) -> None:
    _log(LogLevel.ERROR, message, *args)
