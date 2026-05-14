from typing import Literal, NoReturn, cast

OperationMode = Literal[
    "PARALLEL_IMG",
    "SEQUENTIAL_IMG",
    "PARA_SEQUENTIAL_IMG",
]

IMAGE_OPERATION_MODES = frozenset(
    {"PARALLEL_IMG", "SEQUENTIAL_IMG", "PARA_SEQUENTIAL_IMG"},
)


def _raise_unsupported_operation_mode(operation_mode: str) -> NoReturn:
    if operation_mode == "UV_PASS":
        msg = (
            "Operation mode 'UV_PASS' is not supported. Dedicated UV mode is "
            "currently disabled."
        )
    else:
        msg = f"Operation mode {operation_mode!r} is not supported."
    raise ValueError(msg)


def validate_operation_mode(operation_mode: str) -> OperationMode:
    if operation_mode in IMAGE_OPERATION_MODES:
        return cast("OperationMode", operation_mode)

    _raise_unsupported_operation_mode(operation_mode)


def is_image_operation_mode(operation_mode: str) -> bool:
    return operation_mode in IMAGE_OPERATION_MODES
