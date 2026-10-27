"""Regression test: the DIFFUSION_MODEL wrapper must accept the exact positional
call shape that comfy/ldm/krea2/model.py uses.

Upstream commit c9602625 inserted a new ``ref_latents`` positional argument between
``attention_mask`` and ``transformer_options`` in SingleStreamDiT.forward:

    .execute(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

WrapperExecutor forwards these positionally as ``wrapper(self, x, timesteps, context,
attention_mask, ref_latents, transformer_options)`` -> 7 positional args. A wrapper that
predates the change only has 6 slots and dies with:

    TypeError: krea2t_enhancer_wrapper() takes from 4 to 6 positional arguments but 7 were given

The fix mirrors the one in comfyui-krea2edit: absorb trailing positionals with
``*wargs``, pick ``transformer_options`` as the trailing dict, and pass the exact
positional list through to ``executor``.

Run standalone:  python tests/test_wrapper_signature.py
Or via pytest:   pytest tests/test_wrapper_signature.py
"""
import importlib.util
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)                         # .../custom_nodes/PR-ComfyUI-Krea2T-Enhancer-4f3d93fb
_COMFY_ROOT = os.path.dirname(os.path.dirname(_PACK))  # .../ComfyUI

# ComfyUI root must be importable so the pack's ``import comfy.*`` lines resolve.
if _COMFY_ROOT not in sys.path:
    sys.path.insert(0, _COMFY_ROOT)


def _load_pack():
    """Import the pack's __init__.py under a synthetic name (folder has a hyphen)."""
    spec = importlib.util.spec_from_file_location(
        "krea2t_enhancer_under_test", os.path.join(_PACK, "__init__.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _executor(pe, wrapper, class_obj, sentinel):
    return pe.WrapperExecutor(
        original=lambda *a, **k: sentinel,
        class_obj=class_obj,
        wrappers=[wrapper],
        idx=0,
    )


def test_wrapper_accepts_upstream_positional_call():
    """The crash path: new-style 6-positional call (with ref_latents) must not raise."""
    import comfy.patcher_extension as pe

    mod = _load_pack()
    sentinel = object()
    ex = _executor(pe, mod.krea2t_enhancer_wrapper, object(), sentinel)

    result = ex.execute(
        None,   # x
        None,   # timesteps
        None,   # context
        None,   # attention_mask
        None,   # ref_latents  <-- the argument added by c9602625
        {},     # transformer_options
    )
    assert result is sentinel


def test_wrapper_still_accepts_pre_c9602625_call():
    """Old-style 5-positional call keeps working for pinned older ComfyUI checkouts."""
    import comfy.patcher_extension as pe

    mod = _load_pack()
    sentinel = object()
    ex = _executor(pe, mod.krea2t_enhancer_wrapper, object(), sentinel)

    result = ex.execute(None, None, None, None, {})
    assert result is sentinel


def test_transformer_options_picked_up_under_both_signatures():
    """Behavioral guard: transformer_options must bind to the trailing dict, never to
    ref_latents, under both the old (5-arg) and new (6-arg) positional shapes."""
    import comfy.patcher_extension as pe

    mod = _load_pack()
    real_is_krea2, real_step = mod._is_krea2_dm, mod._step_progress
    mod._is_krea2_dm = lambda dm: True
    captured = {}

    def _record(to):
        captured["to"] = to
        return 0.0, 0.0

    mod._step_progress = _record
    try:
        marker_old = {"tag": "old", "krea2t_prompt_adherence_enhancer": {"enabled": True, "strength": 0.5}}
        marker_new = {"tag": "new", "krea2t_prompt_adherence_enhancer": {"enabled": True, "strength": 0.5}}

        for shape, marker in (("old", marker_old), ("new", marker_new)):
            sentinel = object()
            fake_dm = SimpleNamespace(txtfusion=SimpleNamespace(forward=lambda *a, **k: None))
            ex = _executor(pe, mod.krea2t_enhancer_wrapper, fake_dm, sentinel)
            if shape == "old":
                result = ex.execute(None, None, None, None, marker)            # ...attention_mask, t_opts
            else:
                result = ex.execute(None, None, None, None, ["ref"], marker)   # ...ref_latents, t_opts
            assert result is sentinel, f"{shape}: wrapper did not pass through to executor"
            assert captured["to"] is marker, f"{shape}: got {captured['to']}, expected the trailing dict"
    finally:
        mod._is_krea2_dm, mod._step_progress = real_is_krea2, real_step


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
