"""Contract tests for safe, transparent end-of-training guard."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pulse_repro.train_official import _TEST_END, make_training_entry


def test_guarded_source_is_syntactically_valid(tmp_path: Path):
    src, dest = tmp_path / 'source.py', tmp_path / 'patched.py'
    src.write_text('def main():\n' + _TEST_END + '\n', encoding='utf-8')
    make_training_entry(src, dest)
    text = dest.read_text(encoding='utf-8')
    assert 'if len(test_recs) > 0:' in text
    assert 'final test evaluation skipped' in text
    assert text.count('t = eval_epoch(model, te_dl, device, tta=True)') == 1
    ast.parse(text)


def test_ambiguous_or_missing_anchor_fails_closed(tmp_path: Path):
    src, dest = tmp_path / 'source.py', tmp_path / 'patched.py'
    src.write_text('def main():\n    pass\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='upstream source changed'):
        make_training_entry(src, dest)
    assert not dest.exists()
