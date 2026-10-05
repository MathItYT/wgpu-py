from wgpu.backends.js_webgpu._helpers import descriptor, snake_to_camel


def test_snake_to_camel():
    assert snake_to_camel("required_features") == "requiredFeatures"
    assert snake_to_camel("mip_level_count") == "mipLevelCount"


def test_descriptor_conversion():
    value = descriptor(
        label="triangle",
        required_features=["float32-filterable"],
        required_limits={"max_bind_groups": 4},
        nested={"depth_stencil": {"depth_write_enabled": True}},
        omitted=None,
    )
    assert value == {
        "label": "triangle",
        "requiredFeatures": ["float32-filterable"],
        "requiredLimits": {"maxBindGroups": 4},
        "nested": {"depthStencil": {"depthWriteEnabled": True}},
    }
