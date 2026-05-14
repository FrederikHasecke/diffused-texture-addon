# ruff: noqa: ANN001, FBT001, I001, PLR2004, S101, SLF001

import logging
import sys
from pathlib import Path
from types import SimpleNamespace


def _reset_addon_logger(diagnostics) -> None:
    logger = logging.getLogger("diffused_texture_addon")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if hasattr(logger, diagnostics._LOGGER_CONFIGURED_ATTR):
        delattr(logger, diagnostics._LOGGER_CONFIGURED_ATTR)


def test_setup_logging_keeps_root_logger_unchanged(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from diffused_texture_addon import diagnostics

    _reset_addon_logger(diagnostics)
    monkeypatch.setattr(diagnostics, "_resolve_log_dir", lambda: tmp_path)

    root_logger = logging.getLogger()
    root_handlers = list(root_logger.handlers)
    root_level = root_logger.level

    addon_logger = diagnostics.setup_logging()

    assert root_logger.handlers == root_handlers
    assert root_logger.level == root_level
    assert addon_logger.propagate is False
    assert diagnostics.get_log_file_path() == tmp_path / "diffused_texture_addon.log"


def test_run_stream_logs_subprocess_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from diffused_texture_addon import diagnostics
    from diffused_texture_addon.installer.paths import run_stream

    _reset_addon_logger(diagnostics)
    monkeypatch.setattr(diagnostics, "_resolve_log_dir", lambda: tmp_path)
    diagnostics.setup_logging()

    rc, output = run_stream(
        [
            sys.executable,
            "-c",
            ("import sys; print('alpha'); print('beta'); sys.exit(3)"),
        ],
        label="test stream",
    )

    assert rc == 3
    assert output == "alpha\nbeta"

    recent_logs = diagnostics.get_recent_logs(20)
    assert any("Starting subprocess [test stream]" in entry for entry in recent_logs)
    assert any("[test stream] alpha" in entry for entry in recent_logs)
    assert any("[test stream] beta" in entry for entry in recent_logs)
    assert any(
        "Finished subprocess [test stream] rc=3" in entry for entry in recent_logs
    )


def test_texture_generation_thread_logs_traceback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from diffused_texture_addon import diagnostics
    from diffused_texture_addon import operators

    _reset_addon_logger(diagnostics)
    monkeypatch.setattr(diagnostics, "_resolve_log_dir", lambda: tmp_path)
    diagnostics.setup_logging()

    def _raise_error(*args, **kwargs) -> None:  # noqa: ANN002, ANN003, ARG001
        msg = "boom"
        raise ValueError(msg)

    monkeypatch.setattr(operators, "run_texture_generation", _raise_error)

    results: list[tuple[bool, str | None]] = []

    def _mark_done(
        success: bool,
        error: str | None,
        cancelled: bool = False,
    ) -> None:
        assert cancelled is False
        results.append((success, error))

    operator = SimpleNamespace(_run_id="run123")

    operators.OBJECT_OT_GenerateTexture._run_texture_generation_thread(
        operator,
        process_parameter=SimpleNamespace(operation_mode="PARALLEL_IMG"),
        multiview_images={},
        progress_callback=lambda _value: None,
        should_cancel=lambda: False,
        mark_done=_mark_done,
        return_texture=[],
        input_texture=None,
    )

    assert results == [(False, "boom")]

    recent_logs = diagnostics.get_recent_logs(20)
    assert any(
        "Texture generation thread failed. run_id=run123 mode=PARALLEL_IMG" in entry
        for entry in recent_logs
    )
    assert any("ValueError: boom" in entry for entry in recent_logs)
