import pytest

from model_support import (
    MODEL_SUPPORT_MATRIX,
    get_default_sd_resolution,
    get_sd_version_enum_items,
    require_supported_sd_version,
    supports_ipadapter,
)


def test_supported_model_versions_are_explicit() -> None:
    items = get_sd_version_enum_items()
    assert [i[0] for i in items] == ["sd15", "sdxl"]


def test_unsupported_models_raise_explicit_message() -> None:
    with pytest.raises(ValueError, match="Unsupported model 'flux'"):
        require_supported_sd_version("flux")

    with pytest.raises(ValueError, match="Unsupported model 'qwen'"):
        require_supported_sd_version("qwen")


def test_default_sd_resolution_uses_support_matrix() -> None:
    assert (
        get_default_sd_resolution("sd15", 0)
        == MODEL_SUPPORT_MATRIX["sd15"]["default_sd_resolution"]
    )
    assert (
        get_default_sd_resolution("sdxl", 0)
        == MODEL_SUPPORT_MATRIX["sdxl"]["default_sd_resolution"]
    )
    assert get_default_sd_resolution("sd15", 777) == 777


def test_ipadapter_support_is_matrix_driven() -> None:
    assert supports_ipadapter("sd15") is True
    assert supports_ipadapter("sdxl") is True
    assert supports_ipadapter("flux") is False
    assert supports_ipadapter("qwen") is False
