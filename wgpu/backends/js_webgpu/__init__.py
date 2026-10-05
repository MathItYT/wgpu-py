from __future__ import annotations

from typing import Any

from ... import _classes as classes
from ..._async import GPUPromise
from .. import _register_backend
from ._helpers import descriptor, js_get, js_record_to_dict, make_promise, require_browser_webgpu, snake_to_camel, to_js_value


def _features(obj):
    try: return {str(x) for x in obj.features}
    except (AttributeError, TypeError): return set()


def _limits(obj):
    return js_record_to_dict(js_get(obj, "limits", {}))


def _adapter_info(adapter):
    data = js_record_to_dict(js_get(adapter, "info", {}))
    return classes.GPUAdapterInfo({
        "vendor": data.get("vendor", ""), "architecture": data.get("architecture", ""),
        "device": data.get("device", ""), "description": data.get("description", ""),
        "adapter_type": data.get("adapterType", ""), "backend_type": "webgpu",
    })


class GPU:
    def request_adapter_sync(self, **parameters):
        raise NotImplementedError("Synchronous API functions are unavailable in Pyodide.")

    def request_adapter_async(self, *, feature_level="core", power_preference=None,
                              force_fallback_adapter=False, canvas=None):
        gpu = require_browser_webgpu()
        options = {}
        if power_preference is not None: options["powerPreference"] = power_preference
        if force_fallback_adapter: options["forceFallbackAdapter"] = True
        if feature_level == "compatibility": options["featureLevel"] = feature_level
        if canvas is not None: options["canvas"] = getattr(canvas, "_internal", canvas)
        return make_promise("request_adapter", gpu.requestAdapter(to_js_value(options)),
                            handler=lambda obj: GPUAdapter(obj, _features(obj), _limits(obj), _adapter_info(obj)))

    def get_preferred_canvas_format(self):
        return str(require_browser_webgpu().getPreferredCanvasFormat())

    def get_canvas_context(self, present_info):
        return GPUCanvasContext(present_info)

    @property
    def wgsl_language_features(self):
        try: return {str(x) for x in require_browser_webgpu().wgslLanguageFeatures}
        except (AttributeError, TypeError): return set()


class GPUAdapter(classes.GPUAdapter):
    def __init__(self, internal, features, limits, adapter_info):
        self._internal, self._features, self._limits, self._adapter_info = internal, features, limits, adapter_info

    def request_device_sync(self, **kwargs):
        raise NotImplementedError("Synchronous API functions are unavailable in Pyodide.")

    def request_device_async(self, *, label="", required_features=(), required_limits=None, default_queue=None):
        desc = descriptor(label=label, required_features=required_features, required_limits=required_limits or {})
        return make_promise("request_device", self._internal.requestDevice(to_js_value(desc)),
                            handler=lambda obj: _make_device(obj, self))


def _make_device(obj, adapter):
    queue = GPUQueue("", obj.queue, None)
    return GPUDevice("", obj, adapter, _features(obj), _limits(obj), queue)


class _Base(classes.GPUObjectBase):
    def _release(self): self._internal = None


class GPUDevice(_Base, classes.GPUDevice):
    def __init__(self, label, internal, adapter, features, limits, queue):
        classes.GPUDevice.__init__(self, label, internal, adapter, features, limits, queue)

    def _call(self, method, **kwargs):
        return getattr(self._internal, method)(to_js_value(descriptor(**kwargs)))

    def create_buffer(self, *, label="", size, usage, mapped_at_creation=False):
        obj = self._call("createBuffer", label=label, size=size, usage=usage, mapped_at_creation=mapped_at_creation)
        return GPUBuffer(label, obj, self, size, usage, "mapped" if mapped_at_creation else "unmapped")

    def create_texture(self, *, label="", size, mip_level_count=1, sample_count=1, dimension="2d",
                       format, usage, view_formats=(), texture_binding_view_dimension=None):
        size = to_js_value(size)
        if isinstance(size, (tuple, list)):
            size = {"width": size[0], "height": size[1], "depthOrArrayLayers": size[2] if len(size) > 2 else 1}
        obj = self._call("createTexture", label=label, size=size, mip_level_count=mip_level_count,
                         sample_count=sample_count, dimension=dimension, format=format, usage=usage,
                         view_formats=view_formats, texture_binding_view_dimension=texture_binding_view_dimension)
        info = {"size": (size["width"], size["height"], size["depthOrArrayLayers"]), "mip_level_count": mip_level_count,
                "sample_count": sample_count, "dimension": dimension, "format": format, "usage": usage,
                "texture_binding_view_dimension": texture_binding_view_dimension}
        return GPUTexture(label, obj, self, info)

    def create_sampler(self, **kwargs):
        return GPUSampler(kwargs.get("label", ""), self._call("createSampler", **kwargs), self)

    def create_bind_group_layout(self, *, label="", entries):
        return GPUBindGroupLayout(label, self._call("createBindGroupLayout", label=label, entries=entries), self)

    def create_bind_group(self, *, label="", layout, entries):
        return GPUBindGroup(label, self._call("createBindGroup", label=label, layout=layout, entries=entries), self)

    def create_pipeline_layout(self, *, label="", bind_group_layouts, immediate_size=0):
        return GPUPipelineLayout(label, self._call("createPipelineLayout", label=label, bind_group_layouts=bind_group_layouts, immediate_size=immediate_size), self)

    def create_shader_module(self, *, label="", code, compilation_hints=()):
        return GPUShaderModule(label, self._call("createShaderModule", label=label, code=code), self)

    def create_compute_pipeline(self, *, label="", layout, compute):
        return GPUComputePipeline(label, self._call("createComputePipeline", label=label, layout=layout, compute=compute), self)

    def create_render_pipeline(self, *, label="", layout, vertex, primitive=None, depth_stencil=None, multisample=None, fragment=None):
        return GPURenderPipeline(label, self._call("createRenderPipeline", label=label, layout=layout, vertex=vertex, primitive=primitive, depth_stencil=depth_stencil, multisample=multisample, fragment=fragment), self)

    def create_compute_pipeline_async(self, *, label="", layout, compute):
        return make_promise("create_compute_pipeline", self._internal.createComputePipelineAsync(to_js_value(descriptor(label=label, layout=layout, compute=compute))), handler=lambda x: GPUComputePipeline(label, x, self))

    def create_render_pipeline_async(self, *, label="", layout, vertex, primitive=None, depth_stencil=None, multisample=None, fragment=None):
        return make_promise("create_render_pipeline", self._internal.createRenderPipelineAsync(to_js_value(descriptor(label=label, layout=layout, vertex=vertex, primitive=primitive, depth_stencil=depth_stencil, multisample=multisample, fragment=fragment))), handler=lambda x: GPURenderPipeline(label, x, self))

    def create_command_encoder(self, *, label=""):
        return GPUCommandEncoder(label, self._call("createCommandEncoder", label=label), self)

    def create_render_bundle_encoder(self, *, label="", color_formats, depth_stencil_format=None, sample_count=1, depth_read_only=False, stencil_read_only=False):
        return GPURenderBundleEncoder(label, self._call("createRenderBundleEncoder", label=label, color_formats=color_formats, depth_stencil_format=depth_stencil_format, sample_count=sample_count, depth_read_only=depth_read_only, stencil_read_only=stencil_read_only), self)

    def create_query_set(self, *, label="", type, count):
        return GPUQuerySet(label, self._call("createQuerySet", label=label, type=type, count=count), self, type, count)

    def destroy(self): self._internal.destroy()

    def _get_lost_async(self):
        return make_promise("device_lost", self._internal.lost, handler=lambda x: classes.GPUDeviceLostInfo(str(js_get(x, "reason", "")), str(js_get(x, "message", ""))))


class GPUBuffer(_Base, classes.GPUBuffer):
    def __init__(self, label, internal, device, size, usage, map_state):
        classes.GPUBuffer.__init__(self, label, internal, device, size, usage, map_state)
    def map_async(self, mode, offset=0, size=None):
        size = self._size - offset if size is None else size
        return make_promise("buffer_map", self._internal.mapAsync(mode, offset, size), handler=lambda _: self._set_mapped())
    def _set_mapped(self): self._map_state = "mapped"; return None
    def unmap(self): self._internal.unmap(); self._map_state = "unmapped"
    def read_mapped(self, buffer_offset=None, size=None, *, copy=True):
        offset = 0 if buffer_offset is None else buffer_offset
        size = self._size - offset if size is None else size
        data = self._internal.getMappedRange(offset, size)
        return memoryview(bytes(data.to_py() if hasattr(data, "to_py") else data))
    def write_mapped(self, data, buffer_offset=None):
        from js import Uint8Array
        offset = 0 if buffer_offset is None else buffer_offset
        Uint8Array.new(self._internal.getMappedRange(offset, len(data))).set(memoryview(data).cast("B"))
    def destroy(self): self._internal.destroy()


class GPUTexture(_Base, classes.GPUTexture):
    def __init__(self, label, internal, device, info): classes.GPUTexture.__init__(self, label, internal, device, info)
    def create_view(self, **kwargs):
        return GPUTextureView(kwargs.get("label", ""), self._internal.createView(to_js_value(descriptor(**kwargs))), self._device, self, self._size)

    def destroy(self): self._internal.destroy()


class GPUTextureView(_Base, classes.GPUTextureView):
    def __init__(self, label, internal, device, texture, size):
        classes.GPUTextureView.__init__(self, label, internal, device, texture, size)
class GPUSampler(_Base, classes.GPUSampler): pass
class GPUBindGroupLayout(_Base, classes.GPUBindGroupLayout): pass
class GPUBindGroup(_Base, classes.GPUBindGroup): pass
class GPUPipelineLayout(_Base, classes.GPUPipelineLayout): pass


class GPUShaderModule(_Base, classes.GPUShaderModule):
    def get_compilation_info_async(self):
        return make_promise("shader_compilation_info", self._internal.getCompilationInfo(), handler=lambda x: _CompilationInfo(x))


class _CompilationInfo(classes.GPUCompilationInfo):
    def __init__(self, internal): self._internal = internal
    @property
    def messages(self): return [_CompilationMessage(x) for x in js_get(self._internal, "messages", [])]


class _CompilationMessage(classes.GPUCompilationMessage):
    def __init__(self, internal): self._internal = internal
    message = property(lambda s: str(js_get(s._internal, "message", "")))
    type = property(lambda s: str(js_get(s._internal, "type", "")))
    line_num = property(lambda s: int(js_get(s._internal, "lineNum", 0)))
    line_pos = property(lambda s: int(js_get(s._internal, "linePos", 0)))
    offset = property(lambda s: int(js_get(s._internal, "offset", 0)))
    length = property(lambda s: int(js_get(s._internal, "length", 0)))


class _PipelineBase(_Base):
    def get_bind_group_layout(self, index):
        return GPUBindGroupLayout("", self._internal.getBindGroupLayout(index), self._device)


class GPUComputePipeline(_PipelineBase, classes.GPUComputePipeline): pass
class GPURenderPipeline(_PipelineBase, classes.GPURenderPipeline): pass
class GPUCommandBuffer(_Base, classes.GPUCommandBuffer): pass


class GPUCommandEncoder(_Base, classes.GPUCommandEncoder):
    def begin_compute_pass(self, *, label="", timestamp_writes=None):
        return GPUComputePassEncoder(label, self._internal.beginComputePass(to_js_value(descriptor(label=label, timestamp_writes=timestamp_writes))), self._device)

    def begin_render_pass(self, *, label="", color_attachments, depth_stencil_attachment=None, occlusion_query_set=None, timestamp_writes=None, max_draw_count=50000000):
        return GPURenderPassEncoder(label, self._internal.beginRenderPass(to_js_value(descriptor(
            label=label, color_attachments=color_attachments, depth_stencil_attachment=depth_stencil_attachment,
            occlusion_query_set=occlusion_query_set, timestamp_writes=timestamp_writes, max_draw_count=max_draw_count
        ))), self._device)

    def copy_buffer_to_buffer(self, source, source_offset, destination, destination_offset, size):
        self._internal.copyBufferToBuffer(to_js_value(source), source_offset, to_js_value(destination), destination_offset, size)

    def copy_buffer_to_texture(self, source, destination, copy_size):
        self._internal.copyBufferToTexture(to_js_value(source), to_js_value(destination), to_js_value(copy_size))

    def copy_texture_to_buffer(self, source, destination, copy_size):
        self._internal.copyTextureToBuffer(to_js_value(source), to_js_value(destination), to_js_value(copy_size))

    def copy_texture_to_texture(self, source, destination, copy_size):
        self._internal.copyTextureToTexture(to_js_value(source), to_js_value(destination), to_js_value(copy_size))

    def clear_buffer(self, buffer, offset=0, size=None):
        args = [to_js_value(buffer), offset]
        if size is not None:
            args.append(size)
        self._internal.clearBuffer(*args)

    def resolve_query_set(self, query_set, first_query, query_count, destination, destination_offset):
        self._internal.resolveQuerySet(to_js_value(query_set), first_query, query_count, to_js_value(destination), destination_offset)

    def write_timestamp(self, query_set, query_index):
        self._internal.writeTimestamp(to_js_value(query_set), query_index)

    def finish(self, *, label=""):
        return GPUCommandBuffer(label, self._internal.finish(to_js_value(descriptor(label=label))), self._device)


class _Pass(_Base):
    def set_pipeline(self, pipeline):
        self._internal.setPipeline(to_js_value(pipeline))

    def set_bind_group(self, index, bind_group, dynamic_offsets_data=(), dynamic_offsets_data_start=None, dynamic_offsets_data_length=None):
        offsets = to_js_value(dynamic_offsets_data)
        if dynamic_offsets_data_start is not None:
            start = dynamic_offsets_data_start
            end = None if dynamic_offsets_data_length is None else start + dynamic_offsets_data_length
            offsets = offsets[start:end]
        self._internal.setBindGroup(index, to_js_value(bind_group), offsets)

    def set_immediates(self, range_offset, data, data_offset=0, data_size=None):
        size = len(memoryview(data).cast("B")) - data_offset if data_size is None else data_size
        raw = memoryview(data).cast("B")[data_offset:data_offset + size]
        self._internal.setImmediates(range_offset, raw)

    def push_debug_group(self, group_label):
        self._internal.pushDebugGroup(group_label)

    def pop_debug_group(self):
        self._internal.popDebugGroup()

    def insert_debug_marker(self, marker_label):
        self._internal.insertDebugMarker(marker_label)

    def set_index_buffer(self, buffer, index_format, offset=0, size=None):
        args = [to_js_value(buffer), index_format, offset]
        if size is not None:
            args.append(size)
        self._internal.setIndexBuffer(*args)

    def set_vertex_buffer(self, slot, buffer, offset=0, size=None):
        args = [slot, to_js_value(buffer), offset]
        if size is not None:
            args.append(size)
        self._internal.setVertexBuffer(*args)

    def draw(self, vertex_count, instance_count=1, first_vertex=0, first_instance=0):
        self._internal.draw(vertex_count, instance_count, first_vertex, first_instance)

    def draw_indexed(self, index_count, instance_count=1, first_index=0, base_vertex=0, first_instance=0):
        self._internal.drawIndexed(index_count, instance_count, first_index, base_vertex, first_instance)

    def draw_indirect(self, indirect_buffer, indirect_offset):
        self._internal.drawIndirect(to_js_value(indirect_buffer), indirect_offset)

    def draw_indexed_indirect(self, indirect_buffer, indirect_offset):
        self._internal.drawIndexedIndirect(to_js_value(indirect_buffer), indirect_offset)

    def set_viewport(self, x, y, width, height, min_depth, max_depth):
        self._internal.setViewport(x, y, width, height, min_depth, max_depth)

    def set_scissor_rect(self, x, y, width, height):
        self._internal.setScissorRect(x, y, width, height)

    def set_blend_constant(self, color):
        self._internal.setBlendConstant(to_js_value(color))

    def set_stencil_reference(self, reference):
        self._internal.setStencilReference(reference)

    def execute_bundles(self, bundles):
        self._internal.executeBundles(to_js_value(bundles))

    def begin_occlusion_query(self, query_index):
        self._internal.beginOcclusionQuery(query_index)

    def end_occlusion_query(self):
        self._internal.endOcclusionQuery()

    def dispatch_workgroups(self, x, y=1, z=1):
        self._internal.dispatchWorkgroups(x, y, z)

    def dispatch_workgroups_indirect(self, indirect_buffer, indirect_offset):
        self._internal.dispatchWorkgroupsIndirect(to_js_value(indirect_buffer), indirect_offset)

    def end(self):
        self._internal.end()


class GPUComputePassEncoder(_Pass, classes.GPUComputePassEncoder):
    pass


class GPURenderPassEncoder(_Pass, classes.GPURenderPassEncoder):
    pass


class GPURenderBundle(_Base, classes.GPURenderBundle): pass
class GPURenderBundleEncoder(_Pass, classes.GPURenderBundleEncoder):
    def finish(self, *, label=""): return GPURenderBundle(label, self._internal.finish(to_js_value(descriptor(label=label))), self._device)


class GPUQueue(_Base, classes.GPUQueue):
    def submit(self, command_buffers): self._internal.submit(to_js_value(command_buffers))
    def write_buffer(self, buffer, buffer_offset, data, data_offset=0, size=None):
        raw = memoryview(data).cast("B"); end = raw.nbytes if size is None else data_offset + size
        self._internal.writeBuffer(to_js_value(buffer), buffer_offset, raw[data_offset:end])
    def write_texture(self, destination, data, data_layout, size):
        self._internal.writeTexture(to_js_value(destination), memoryview(data).cast("B"), to_js_value(data_layout), to_js_value(size))

    def read_buffer(self, buffer, buffer_offset=0, size=None):
        raise NotImplementedError(
            "Synchronous queue.read_buffer() is unavailable in Pyodide; use "
            "GPUBuffer.map_async() after submitting a COPY_SRC buffer."
        )


class GPUQuerySet(_Base, classes.GPUQuerySet):
    def destroy(self): self._internal.destroy()


gpu = GPU()
_register_backend(gpu)



class GPUCanvasContext:
    """rendercanvas-facing WebGPU presentation context.

    This object intentionally mirrors the small context API expected by
    rendercanvas.WgpuContextToScreen while delegating presentation to the
    browser's native GPUCanvasContext.
    """

    def __init__(self, present_info):
        self._present_info = present_info
        self._canvas = present_info["window"]
        self._context = self._canvas.getContext("webgpu")
        if self._context is None:
            raise RuntimeError("The supplied canvas does not provide a WebGPU context.")
        self._config = None
        self._physical_size = (0, 0)

    def set_physical_size(self, width, height):
        self._physical_size = (int(width), int(height))

    @property
    def physical_size(self):
        return self._physical_size

    def get_preferred_format(self, adapter=None):
        return str(require_browser_webgpu().getPreferredCanvasFormat())

    def configure(self, *, device, format=None, usage=0x10, view_formats=(), alpha_mode="opaque", **kwargs):
        if format is None:
            format = self.get_preferred_format(getattr(device, "adapter", None))

        if isinstance(usage, str):
            import wgpu
            bits = 0
            for name in usage.replace("|", " ").split():
                bits |= wgpu.TextureUsage[name]
            usage = bits

        js_config = {
            "device": to_js_value(device),
            "format": format,
            "usage": usage,
            "viewFormats": list(view_formats),
            "alphaMode": alpha_mode,
        }

        # WebGPU canvas colorSpace/toneMapping are browser features and should
        # only be sent when explicitly requested.
        if "color_space" in kwargs and kwargs["color_space"] is not None:
            js_config["colorSpace"] = kwargs["color_space"]
        if "tone_mapping" in kwargs and kwargs["tone_mapping"] is not None:
            js_config["toneMapping"] = to_js_value(kwargs["tone_mapping"])

        self._context.configure(to_js_value(js_config))
        self._config = {
            "device": device,
            "format": format,
            "usage": usage,
            "view_formats": tuple(view_formats),
            "alpha_mode": alpha_mode,
        }

    def unconfigure(self):
        self._context.unconfigure()
        self._config = None

    def get_current_texture(self):
        if self._config is None:
            raise RuntimeError("Canvas context must be configured before calling get_current_texture().")

        texture = self._context.getCurrentTexture()
        width, height = self._physical_size

        # Browser canvas dimensions are authoritative when rendercanvas has not
        # supplied a physical size yet.
        if width <= 0 or height <= 0:
            width = int(self._canvas.width)
            height = int(self._canvas.height)

        info = {
            "size": (width, height, 1),
            "mip_level_count": 1,
            "sample_count": 1,
            "dimension": "2d",
            "format": self._config["format"],
            "usage": self._config["usage"],
            "texture_binding_view_dimension": None,
        }
        return GPUTexture("current_canvas_texture", texture, self._config["device"], info)

    def present(self):
        # Unlike native window backends, browser WebGPU has no swapBuffers().
        # Submitting commands is enough; the configured canvas texture is
        # composited by the browser.
        return None

    def _release(self):
        try:
            self.unconfigure()
        except Exception:
            pass
        self._context = None
        self._canvas = None
        self._config = None



# Instantiate and register this backend. Pyodide selects this module automatically.
gpu = GPU()
_register_backend(gpu)
