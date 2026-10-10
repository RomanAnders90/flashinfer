# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Unified Frost MoELayer configuration, admission, and GPU integration tests."""

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import replace
import importlib
from types import SimpleNamespace
from unittest.mock import Mock
import warnings

import pytest
import torch

from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import (
    BackendOptions,
    CudnnFrostBf16Config,
    CudnnFrostMxfp8Config,
    CudnnFrostMxfp8Mxfp4Config,
    CudnnFrostNvfp4Config,
    CutlassBf16Config,
    CutlassMxfp8Config,
    CutlassMxfp8Mxfp4Config,
    CutlassNvfp4Config,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
)
from flashinfer.fused_moe import api, layer as layer_module
from flashinfer.fused_moe.auto_candidates import _AUTO_CANDIDATES
from flashinfer.fused_moe.backends.cudnn_frost import compiler
from tests.moe.test_cudnn_frost_kernels import (
    _bf16_moe_reference,
    _mxfp8_moe_reference,
    _nvfp4_moe_reference,
)

FORMATS = {
    "bf16": (
        CudnnFrostBf16Config,
        CutlassBf16Config,
        QuantFormat.BF16,
        QuantFormat.BF16,
    ),
    "mxfp8": (
        CudnnFrostMxfp8Config,
        CutlassMxfp8Config,
        QuantFormat.MXFP8,
        QuantFormat.MXFP8,
    ),
    "nvfp4": (
        CudnnFrostNvfp4Config,
        CutlassNvfp4Config,
        QuantFormat.NVFP4,
        QuantFormat.NVFP4,
    ),
    "mxfp8_mxfp4": (
        CudnnFrostMxfp8Mxfp4Config,
        CutlassMxfp8Mxfp4Config,
        QuantFormat.MXFP4,
        QuantFormat.MXFP8,
    ),
}


@pytest.mark.parametrize(
    "config,key",
    [
        (CudnnFrostBf16Config, "cudnn_frost_bf16"),
        (CudnnFrostMxfp8Config, "cudnn_frost_mxfp8"),
        (CudnnFrostNvfp4Config, "cudnn_frost_nvfp4"),
        (CudnnFrostMxfp8Mxfp4Config, "cudnn_frost_mxfp8_mxfp4"),
    ],
)
def test_public_config_registration(config, key):
    assert config in api.ALL_BACKEND_CONFIGS
    assert eval(repr(config()), vars(api)) == config()
    assert BackendOptions((config(),)).valid_for(107) == [config()]
    assert BackendOptions((config(),)).valid_for(120) == (
        [config()] if config is CudnnFrostBf16Config else []
    )
    assert BackendOptions((config(),)).valid_for(121) == []
    assert BackendOptions((config(),)).valid_for(100) == []
    assert layer_module._BACKEND_RUNNERS[config].backend_key == key
    assert not _AUTO_CANDIDATES[key].experimental


@pytest.mark.parametrize(
    "module",
    ["cache", "nvfp4.moe"],
)
def test_legacy_import_shares_module_and_caches(module):
    old = "flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm"
    new = "flashinfer.fused_moe.backends.cudnn_frost"
    assert importlib.import_module(f"{old}.{module}") is importlib.import_module(
        f"{new}.{module}"
    )


def test_frost_weight_view_prefers_own_key_and_reuses_canonical(monkeypatch):
    from flashinfer.fused_moe.auto_candidates import additional_candidates
    from flashinfer.fused_moe.backends.cudnn_frost.support import weight_view

    canonical, own = {"canonical": torch.empty(0)}, {"own": torch.empty(0)}
    weights = MoEWeightPack({"cutlass_nvfp4": canonical})
    assert weight_view(weights, "cudnn_frost_nvfp4", "cutlass_nvfp4") is canonical
    weights.prepare_for("cudnn_frost_nvfp4", own)
    assert weight_view(weights, "cudnn_frost_nvfp4", "cutlass_nvfp4") is own

    key = "cudnn_frost_nvfp4"
    support = importlib.import_module(_AUTO_CANDIDATES[key].support_module)
    runner = SimpleNamespace(backend_key=key, accepts=lambda *args: True)
    create = Mock(return_value=runner)
    monkeypatch.setattr(support, "is_eligible", lambda *args: True)
    monkeypatch.setattr(support, "create_runner", create)
    cache = {}
    options = dict(tuning=True, cache=cache, exclude=set(_AUTO_CANDIDATES) - {key})
    weights.native_views.clear()
    assert additional_candidates(None, None, 107, None, weights, **options) == []
    assert not create.called
    for view_key, view in (("cutlass_nvfp4", canonical), (key, own)):
        weights.prepare_for(view_key, view)
        assert additional_candidates(None, None, 107, None, weights, **options) == [
            runner
        ]
        weights.native_views.clear()
    assert create.call_count == 1
    assert additional_candidates(None, None, 107, None, weights, **options) == []
    assert cache[key] is runner  # Older packs/graphs keep the cached resource.


def test_explicit_frost_uses_standard_packing_and_reuses_token_bucket(monkeypatch):
    class Runner:
        supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)

        def __init__(self, key, requires_view):
            self.backend_key, self.requires_view = key, requires_view
            self.generation = 0
            self.validations = 0
            self.packings = 0

        def accepts(self, act, weights):
            return True

        def refresh(self):
            self.generation += 1

        def pack_inputs(self, act, weights):
            self.validations += 1
            if self.requires_view:
                weights.native_views[self.backend_key]
            self.packings += 1
            return [act.hidden_states_q]

        def launch_kwargs_for(self, inputs):
            return {}

        def forward(self, inputs, **kwargs):
            return inputs[0]

    config = MoEConfig(
        routing=RoutingConfig(num_experts=64, top_k=6),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=1408),
        backend=BackendOptions((CudnnFrostBf16Config(),)),
    )
    frost = Runner("cudnn_frost_bf16", True)
    fallback = Runner("fallback", False)
    layer = MoELayer.__new__(MoELayer)
    layer.config, layer.device, layer._arch = config, torch.device("cuda:0"), 107
    layer.runners, layer._automatic_runners, layer._winners = (
        [fallback, frost],
        {},
        OrderedDict(),
    )
    layer.tuner = SimpleNamespace(is_tuning_mode=True)
    support = importlib.import_module(
        _AUTO_CANDIDATES[frost.backend_key].support_module
    )
    monkeypatch.setattr(
        support, "create_runner", lambda *args: pytest.fail("duplicate Frost runner")
    )
    monkeypatch.setattr(layer_module, "map_to_hybrid_bucket", lambda *args: 16)
    selections = []

    def select(act, weights, runners):
        selections.append((act.num_tokens, tuple(r.backend_key for r in runners)))
        return runners[-1], -1

    monkeypatch.setattr(layer, "_select_winner", select)
    from flashinfer import api_logging

    monkeypatch.setattr(
        api_logging,
        "warn_experimental_backend_once",
        lambda *args: pytest.fail("stable warning"),
    )
    monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", raising=False)
    weights = MoEWeightPack({"cudnn_frost_bf16": {}})

    def act(tokens):
        return MoEActivationPack(
            torch.empty(tokens, 2048, device="meta", dtype=torch.bfloat16),
            None,
            torch.empty(tokens, 6, device="meta", dtype=torch.int32),
            torch.empty(tokens, 6, device="meta", dtype=torch.float32),
        )

    a, b = act(9), act(10)
    assert layer(a, weights) is a.hidden_states_q
    assert layer(b, weights) is b.hidden_states_q
    assert layer.winner_backend == frost.backend_key
    assert len(selections) == 1  # Same bucket reuses its backend and tactic.
    weights.native_views.clear()
    with pytest.raises(KeyError, match="cudnn_frost_bf16"):
        layer(a, weights)
    assert len(selections) == 1  # Invalid packs fail at packing, without fallback.
    weights.prepare_for("cudnn_frost_bf16", {})
    layer(a, weights)
    assert layer.winner_backend == frost.backend_key
    assert len(selections) == 1
    assert not layer._automatic_runners
    assert frost.validations == 4
    assert frost.packings == 3
    layer(a, weights)
    assert len(selections) == 1
    assert frost.validations == 5
    assert frost.packings == 4
    frost.refresh()
    layer(a, weights)
    assert len(selections) == 1  # Runner snapshots do not qualify layer winners.
    assert frost.validations == 6
    assert frost.packings == 5
    layer.refresh_backends()
    assert not layer._winners
    layer(a, weights)
    assert len(selections) == 2  # Explicit layer refresh clears bucket winners.
    assert frost.validations == 7
    assert frost.packings == 6


@pytest.mark.parametrize(
    "version,accepted",
    [
        ("release 12.9, V12.9.86", False),
        ("release 13.3, V13.3.59", False),
        ("release 13.4", True),
        ("release 13.4, V13.4.59", True),
        ("release 13.4, V13.4.91", True),
        ("release 13.4, V13.4.92", True),
        ("release 13.5, V13.5.23", True),
        ("release 14.0, V14.0.1", True),
        ("unknown", False),
    ],
)
def test_selected_external_assembler_release(version, accepted, monkeypatch):
    monkeypatch.setattr(
        compiler,
        "compiler_identity",
        lambda: dict(
            backend="external_ptxas",
            version=f"Cuda compilation tools, {version}",
        ),
    )
    if accepted:
        compiler.require_moe_assembler()
    else:
        with pytest.raises(NotImplementedError, match="CUDA 13.4"):
            compiler.require_moe_assembler()


@pytest.mark.parametrize(
    "version,accepted",
    [
        ((12, 9), False),
        ((13, 3), False),
        ((13, 4), True),
        ((13, 5), True),
        ((14, 0), True),
    ],
)
def test_bundled_assembler_is_independent_of_cuda_home(version, accepted, monkeypatch):
    cutlass = pytest.importorskip("cutlass")
    monkeypatch.setattr(compiler, "compiler_identity", lambda: dict(backend="bundled"))
    monkeypatch.setattr(
        cutlass,
        "CUDA_VERSION",
        SimpleNamespace(major=version[0], minor=version[1]),
        raising=False,
    )
    monkeypatch.setenv("CUDA_HOME", "/unused/cuda-14.0")
    if accepted:
        compiler.require_moe_assembler()
    else:
        with pytest.raises(NotImplementedError, match="CUDA 13.4"):
            compiler.require_moe_assembler()


@pytest.mark.parametrize(
    "dtype,name",
    [
        ("bf16", "CudnnFrostBf16MoeRunner"),
        ("mxfp8", "CudnnFrostMxfp8MoeRunner"),
        ("nvfp4", "CudnnFrostNvfp4MoeRunner"),
        ("mxfp8_mxfp4", "CudnnFrostMxfp8Mxfp4MoeRunner"),
    ],
)
def test_runner_refresh_rechecks_selected_assembler(dtype, name, monkeypatch):
    from flashinfer.fused_moe.backends.cudnn_frost import engine

    module = importlib.import_module(
        f"flashinfer.fused_moe.backends.cudnn_frost.{dtype}.moe"
    )
    runner = object.__new__(getattr(module, name))
    runner.device, runner.config, runner._built = torch.device("cuda:0"), None, True
    selected = dict(backend="external_ptxas", version="release 13.5")
    monkeypatch.setattr(engine.runtime, "_arch_for", lambda _: "sm_107a")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=1),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reads = Mock(side_effect=lambda: selected.copy())
    monkeypatch.setattr(compiler, "_read_compiler_identity", reads)
    kernel = SimpleNamespace(tactic="test")
    runner._engine = engine.FrostEngine(
        runner.device,
        artifact_roots=lambda: (),
        artifact_version=lambda: 0,
        refresh_artifacts=lambda: None,
        policy_version="test",
        tactic_tag="test",
        select_kernels=lambda _: ((kernel,), (kernel,)),
        select_fma=lambda _: None,
        build_plan=lambda *args: pytest.fail("admission must not build a plan"),
        require_assembler=True,
    )
    validation = Mock(side_effect=ValueError("invalid pack"))
    monkeypatch.setattr(runner, "pack_inputs", validation)
    act, weights = object(), object()
    assert runner.accepts(act, weights)
    assert not validation.called
    key = runner._engine._snapshot_key
    with pytest.raises(ValueError, match="invalid pack"):
        runner.pack_inputs(act, weights)
    assert validation.call_count == 1

    selected["version"] = "release 13.3"
    assert runner.accepts(act, weights)
    assert runner._engine._snapshot_key == key
    assert reads.call_count == 1  # Admission leaves the built snapshot unchanged.
    with pytest.raises(NotImplementedError, match="CUDA 13.4"):
        runner.refresh()
    assert runner._engine._snapshot_key == key
    selected["version"] = "release 13.6"
    runner.refresh()
    assert runner._engine._snapshot_key != key
    assert reads.call_count == 3


@pytest.fixture(scope="module", params=FORMATS)
def frost_case(request):
    return _make_frost_case(request.param)


def _make_frost_case(dtype, geometry=(64, 2048, 1408, 6)):
    supported = ((10, 7), (12, 0)) if dtype == "bf16" else ((10, 7),)
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in supported
    ):
        pytest.skip(f"Frost {dtype} requires compute capability in {supported}")
    from flashinfer.fused_moe.backends.cudnn_frost.compiler import require_moe_assembler
    from flashinfer.fused_moe.backends.cudnn_frost.cache import (
        require_graph_resource_retention,
    )

    try:
        if torch.cuda.get_device_capability() == (10, 7):
            require_moe_assembler()
        require_graph_resource_retention()
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    cls, canonical, weight, activation = FORMATS[dtype]
    e, h, i, k = geometry
    config = MoEConfig(
        routing=RoutingConfig(num_experts=e, top_k=k),
        quant=QuantConfig(weight=weight, activation=activation),
        experts=ExpertConfig(intermediate_size=i),
        backend=BackendOptions((cls(),)),
        execution=ExecutionConfig(tune_max_num_tokens=512),
    )
    layer = MoELayer(config)
    torch.manual_seed(41)
    w1 = torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) * 0.02
    view = cls.prepare_weights(
        w1,
        w2,
        num_local_experts=e,
        hidden_size=h,
        intermediate_size=i,
        activation=config.activation,
    )
    return dtype, cls, canonical, config, layer, view


def test_public_config_graph_and_automatic_admission(frost_case, monkeypatch):
    tokens = 9
    dtype, cls, canonical, config, layer, view = frost_case
    monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", raising=False)
    key = "cudnn_frost_" + dtype
    own = MoEWeightPack({key: view})
    shared = MoEWeightPack({"cutlass_" + dtype: view})
    x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
    xq, xsf = cls.prepare_activations(x, quant=config.quant)
    # Include idle experts and a skewed distribution.
    ids = torch.randint(0, 32, (tokens, 6), dtype=torch.int32, device="cuda")
    scores = torch.rand(tokens, 6, device="cuda").softmax(-1)
    act = MoEActivationPack(xq, xsf, ids, scores)

    def reference():
        if dtype == "bf16":
            return _bf16_moe_reference(act, shared, config.activation)
        if dtype == "nvfp4":
            return _nvfp4_moe_reference(act, shared, config.activation)
        return _mxfp8_moe_reference(
            act, shared, config.activation, mixed=dtype == "mxfp8_mxfp4"
        )

    def check(out):
        ref = reference()
        assert torch.isfinite(out).all()
        error = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
        assert error < (0.005 if dtype == "bf16" else 0.03), error

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with autotune(tuning_buckets=(tokens,)):
            out = layer(act, own).clone()
        check(out)
        assert layer.winner_backend == key
        assert not layer._automatic_runners
        torch.testing.assert_close(layer(act, shared), out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = layer(act, own)
        ids.fill_(63)
        graph_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(graph_out)

        layer.refresh_backends()
        assert not layer._winners
        assert layer.winner_backend is None
        check(layer(act, own))
        graph_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(graph_out)

        automatic = MoELayer(replace(config, backend=BackendOptions((canonical(),))))

        def select_frost(act, weights, runners):
            # Verify admission and real execution independent of performance.
            runner = next(r for r in runners if r.backend_key == key)
            packed = runner.pack_inputs(act, weights)
            return runner, runner.get_valid_tactics(packed, None)[0]

        monkeypatch.setattr(automatic, "_select_winner", select_frost)
        with autotune(tuning_buckets=(tokens,)):
            check(automatic(act, shared))
        monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "0")
        check(automatic(act, shared))
        assert automatic.winner_backend == key
    assert not any(type(w.message).__name__ == "ExperimentalWarning" for w in caught)


@pytest.mark.parametrize("dtype", ("mxfp8", "nvfp4", "mxfp8_mxfp4"))
def test_quantized_selection_cache(dtype, monkeypatch):
    moe = importlib.import_module(
        f"flashinfer.fused_moe.backends.cudnn_frost.{dtype}.moe"
    )
    props = SimpleNamespace(multi_processor_count=128)
    monkeypatch.setattr(moe.common, "_arch_for", lambda _: "sm_107a")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: props)
    selection = Mock(wraps=moe.select_stages)
    monkeypatch.setattr(moe, "select_stages", selection)
    args = dict(
        tokens=17,
        hidden=2048,
        intermediate=1536,
        experts=16,
        topk=2,
        device="cpu",
        activation=api.SwiGLU(),
    )
    moe.runtime.clear_artifact_cache()
    try:
        first, second = moe._selected_kernels(**args)
        assert sum(not k.quantizes_output for k in first) == len(second) == 4
        for _ in range(25):
            assert moe._selected_kernels(**args) == (first, second)
        assert selection.call_count == 1

        # Use real artifacts for the other admission branch, beyond offline T.
        profiles = moe._read(moe._artifact_roots())[
            ("sm_107a", "swiglu", 64, 2048, 1408, 6)
        ]
        moe._selected_kernels(
            **(
                args
                | dict(tokens=max(profiles) + 1, intermediate=1408, experts=64, topk=6)
            )
        )
        assert selection.call_count == 2
        for changes in (
            dict(tokens=18),
            dict(hidden=4096),
            dict(intermediate=3072),
            dict(experts=32),
            dict(topk=4),
            dict(activation=api.ReLU()),
        ):
            misses = moe._selected_kernels_cached.cache_info().misses
            moe._selected_kernels(**(args | changes))
            assert moe._selected_kernels_cached.cache_info().misses == misses + 1

        for owner, name, value in (
            (props, "multi_processor_count", 120),
            (compiler, "identity_key", lambda: "changed"),
            (moe.common, "_arch_for", lambda _: "sm_120a"),
            (moe, "_artifact_roots", lambda: ()),
        ):
            monkeypatch.setattr(owner, name, value)
            misses = moe._selected_kernels_cached.cache_info().misses
            moe._selected_kernels(**args)
            assert moe._selected_kernels_cached.cache_info().misses == misses + 1

        version = moe.runtime._artifact_cache_version
        moe.runtime.clear_artifact_cache()
        assert moe.runtime._artifact_cache_version == version + 1
        for tokens in range(1, 130):
            assert moe._selected_kernels(**(args | dict(tokens=tokens))) == ((), ())
        info = moe._selected_kernels_cached.cache_info()
        assert info.currsize == info.maxsize == 128
        moe._selected_kernels(**(args | dict(tokens=129)))
        assert moe._selected_kernels_cached.cache_info().misses == info.misses
        moe._selected_kernels(**(args | dict(tokens=1)))
        assert moe._selected_kernels_cached.cache_info().misses == info.misses + 1
    finally:
        moe.runtime.clear_artifact_cache()


@pytest.mark.parametrize("dtype", ("mxfp8", "nvfp4", "mxfp8_mxfp4"))
def test_factorized_tuning_and_graph_replay(dtype, monkeypatch):
    from flashinfer.fused_moe.backends.cudnn_frost import tuning

    tokens, e, h, i, k = 17, 16, 2048, 1536, 2
    moe = importlib.import_module(
        f"flashinfer.fused_moe.backends.cudnn_frost.{dtype}.moe"
    )
    moe.runtime.clear_artifact_cache()
    capture_selection = moe._capture_selection
    selection = Mock()

    def capture(*args):
        snapshot = capture_selection(*args)

        def select(shape):
            selection(shape)
            return snapshot.select_kernels(shape)

        return snapshot._replace(select_kernels=select)

    monkeypatch.setattr(moe, "_capture_selection", capture)
    _, cls, _, config, layer, view = _make_frost_case(dtype, (e, h, i, k))
    stage_calls = set()
    original_forward = tuning._StageRunner.forward

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        stage_calls.add((self.stage, do_preparation))
        return original_forward(self, inputs, tactic, do_preparation, **kwargs)

    monkeypatch.setattr(tuning._StageRunner, "forward", forward)
    weights = MoEWeightPack({"cutlass_" + dtype: view})
    x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16)
    xq, xsf = cls.prepare_activations(x, quant=config.quant)
    ids = torch.rand(tokens, e, device="cuda").topk(k, dim=-1).indices.int()
    scores = torch.rand(tokens, k, device="cuda").softmax(-1).bfloat16().float()
    act = MoEActivationPack(xq, xsf, ids, scores)

    def reference():
        if dtype == "nvfp4":
            return _nvfp4_moe_reference(act, weights, config.activation)
        return _mxfp8_moe_reference(
            act, weights, config.activation, mixed=dtype == "mxfp8_mxfp4"
        )

    def check(out, ref):
        assert torch.isfinite(out).all()
        error = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
        assert error < 0.03, error

    tuner = AutoTuner.get()
    tuner.clear_cache()
    try:
        with autotune(tuning_buckets=(tokens,)):
            out = layer(act, weights).clone()
        assert stage_calls == {(1, True), (1, False), (2, True), (2, False)}
        ref = reference()
        check(out, ref)
        check(layer(act, weights), ref)
        assert selection.call_count == 1  # Packing reuses this bucket's frozen pool.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = layer(act, weights)
        ids.copy_(torch.tensor([e - 2, e - 1], device="cuda", dtype=ids.dtype))
        scores.copy_(scores.flip(-1))
        scale_key = (
            "fc1_weight_block_scale" if dtype == "nvfp4" else "fc1_expert_scales"
        )
        # The captured launch reads caller-owned SF directly; a content update
        # needs neither a new pack nor graph recapture.
        view[scale_key].view(torch.uint8).add_(1)
        changed_ref = reference()
        assert not torch.allclose(ref, changed_ref)
        graph_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(graph_out, changed_ref)
        assert selection.call_count == 1
    finally:
        tuner.clear_cache()
        moe.runtime.clear_artifact_cache()
