"""End-to-end variant contract and source-patching tests, no downloads."""
import copy
from pathlib import Path

import pytest
import torch

from pulse_repro.joint_v1 import PULSESelectiveJointV1
from pulse_repro.selective_v1 import PULSESelectiveV1
from pulse_repro.train_joint_v1 import (
    _replace_once, build_entry, _SAVE_ANCHOR,
    _MODEL_ANCHOR, _IMPORT_ANCHOR, _TEST_END,
)
from test_selective_v1 import MiniPULSE


def test_joint_model_trains_original_and_adapter_weights():
    torch.manual_seed(7)
    base = MiniPULSE()
    base.backbone_blocks = torch.nn.ModuleList([
        torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
    ])
    control = copy.deepcopy(base)
    model = PULSESelectiveJointV1(base)
    model.train()
    assert model.training
    assert model.base.training
    assert all(p.requires_grad for p in model.parameters())
    assert model.backbone_blocks is model.base.backbone_blocks
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        ref = control(x)
        got = model(x)
    for a, b in zip(ref, got):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_joint_weights_load_in_existing_eval_variant():
    model = PULSESelectiveJointV1(MiniPULSE())
    evaluator = PULSESelectiveV1(MiniPULSE())
    evaluator.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        got = model.eval()(x)
        reread = evaluator.eval()(x)
    for a, b in zip(got, reread):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_exact_upstream_patch_and_guard(tmp_path: Path):
    # A minimal fixture with the exact patch anchors, no PULSE network download.
    raw = '\n'.join([
        'from transformers import AutoModel',
        '',
        'def main():',
        _MODEL_ANCHOR,
        "    if True:",
        _SAVE_ANCHOR,
        '    test_recs = []',
        _TEST_END,
        '',
        'if __name__ == "__main__":',
        '    main()',
    ])
    upstream = tmp_path / 'pulse_seg.py'
    entry = tmp_path / 'joint_v1_training_entry.py'
    upstream.write_text(raw, encoding='utf-8')
    build_entry(upstream, entry)
    patched = entry.read_text(encoding='utf-8')
    assert 'PULSESelectiveJointV1(PULSESeg(IMG_SIZE))' in patched
    assert '"variant": "selective-v1"' in patched
    assert '"training_mode": "joint"' in patched
    assert 'if len(test_recs) > 0:' in patched
    assert _MODEL_ANCHOR not in patched
    assert _SAVE_ANCHOR not in patched
    # Original upstream bytes never modified.
    assert upstream.read_text(encoding='utf-8') == raw
    compile(patched, str(entry), 'exec')


def test_unexpected_upstream_refuses_patching(tmp_path: Path):
    upstream = tmp_path / 'pulse_seg.py'
    upstream.write_text('def main(): pass\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='anchor'):
        build_entry(upstream, tmp_path / 'entry.py')
    with pytest.raises(RuntimeError):
        _replace_once('a', 'different', 'b')
