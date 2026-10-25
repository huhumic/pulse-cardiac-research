"""Fast CPU regression tests for non-background-dominated teacher diagnostics."""
import pytest
import torch

from pulse_repro.v3_frozen_teacher import PropagatedLabels
from pulse_repro.v3_teacher_pilot import structural_agreement_diagnostic


def test_per_class_pairwise_dice_reports_disagreement_not_accuracy():
    ed = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).view(1,1,1,2,4)
    es = torch.tensor([0, 1, 1, 2, 2, 2, 3, 0]).view(1,1,1,2,4)
    valid = torch.ones_like(ed, dtype=torch.bool)
    agree = valid & (ed == es)
    result = structural_agreement_diagnostic(
        PropagatedLabels(ed, es, valid, valid, agree,
                         torch.where(agree, ed, torch.full_like(ed, 255)))
    )
    assert result["both_endpoint_valid_voxels"] == 8
    assert result["agreement_all_classes_in_both_valid"] == pytest.approx(5 / 8)
    assert result["foreground_union_voxels"] == 7
    assert result["foreground_agree_voxels"] == 4
    assert result["foreground_agreement_given_either_foreground"] == pytest.approx(4 / 7)
    assert result["classes"]["RV"]["pairwise_dice_not_gt"] == pytest.approx(1/2)
    assert result["classes"]["Myo"]["pairwise_dice_not_gt"] == pytest.approx(4/5)
    assert result["classes"]["LV"]["pairwise_dice_not_gt"] == pytest.approx(2/3)


def test_no_both_valid_or_no_foreground_is_explicit_none():
    a = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
    invalid = torch.zeros_like(a, dtype=torch.bool)
    r = structural_agreement_diagnostic(
        PropagatedLabels(a, a, invalid, invalid, invalid, a)
    )
    assert r["agreement_all_classes_in_both_valid"] is None
    assert r["foreground_agreement_given_either_foreground"] is None
    assert all(v["pairwise_dice_not_gt"] is None for v in r["classes"].values())
