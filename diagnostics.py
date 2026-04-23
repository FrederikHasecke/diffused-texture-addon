import logging
import tempfile
import threading
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "diffused_texture_addon"
_LOGGER_CONFIGURED_ATTR = "_diffusedtexture_configured"
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] [%(threadName)s] %(message)s"
_MAX_BUFFER_LINES = 500
_MAX_LOG_BYTES = 512 * 1024
_BACKUP_COUNT = 3


class _MemoryLogHandler(logging.Handler):
    def __init__(self, max_entries: int = _MAX_BUFFER_LINES) -> None:
        super().__init__(level=logging.DEBUG)
        self._entries: deque[str] = deque(maxlen=max_entries)
        self._entries_lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001
            message = record.getMessage()

        with self._entries_lock:
            self._entries.append(message)

    def tail(self, limit: int) -> list[str]:
        with self._entries_lock:
            return list(self._entries)[-limit:]


def _resolve_log_dir() -> Path:
    try:
        import bpy

        base_dir = Path(
            bpy.utils.user_resource(
                "CONFIG",
                path="diffused_texture_addon",
                create=True,
            ),
        )
    except Exception:  # noqa: BLE001
        base_dir = Path(tempfile.gettempdir()) / "diffused_texture_addon"

    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def setup_logging() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if getattr(logger, _LOGGER_CONFIGURED_ATTR, False):
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    memory_handler = _MemoryLogHandler()
    memory_handler.setFormatter(formatter)
    logger.addHandler(memory_handler)

    log_path: Path | None = None
    try:
        log_path = _resolve_log_dir() / "diffused_texture_addon.log"
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Failed to initialize file logging at %s.",
            log_path,
            exc_info=True,
        )

    setattr(logger, _LOGGER_CONFIGURED_ATTR, True)
    logger.debug(
        "Addon diagnostics initialized. Log file: %s",
        log_path if log_path is not None else "<memory only>",
    )
    return logger


def get_logger(component: str | None = None) -> logging.Logger:
    setup_logging()
    if not component:
        return logging.getLogger(_LOGGER_NAME)

    normalized = component.strip(".")
    if normalized.startswith(f"{_LOGGER_NAME}.") or normalized == _LOGGER_NAME:
        return logging.getLogger(normalized)

    return logging.getLogger(f"{_LOGGER_NAME}.{normalized}")


def get_log_file_path() -> Path | None:
    logger = setup_logging()
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            return Path(handler.baseFilename)
    return None


def get_recent_logs(limit: int = 50) -> list[str]:
    logger = setup_logging()
    for handler in logger.handlers:
        if isinstance(handler, _MemoryLogHandler):
            return handler.tail(limit)
    return []
