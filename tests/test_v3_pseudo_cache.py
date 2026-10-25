"""CPU fast tests: G2 candidate reliability, label ignore and safety contracts."""
import numpy as np
import pytest
import torch

from pulse_repro.v3_frozen_teacher import PropagatedLabels
from pulse_repro.v3_pseudo_cache import (
    class_balanced_pseudolabel_ce, reliability_from_image_residuals,
)
from pulse_repro.v3_uvi_acdc_adapter import TEACHER_SHAPE, split_role


def toy():
    shape = (1,1,*TEACHER_SHAPE)
    a = torch.zeros(shape, dtype=torch.long)
    b = a.clone()
    measured = torch.zeros(shape, dtype=torch.bool)
    measured[:, :, 16:20, 16:20, 11:21] = True
    a[:, :, 16:18, 16:18, 11:21] = 1
    b[:, :, 16:18, 16:18, 11:21] = 1
    a[:, :, 19, 19, 11:21] = 1  # disagree
    agreement = measured & (a == b)
    p = torch.where(agreement, a, torch.full_like(a, 255))
    return PropagatedLabels(a,b,measured,measured,agreement,p)


def test_weight_masks_disagreement_padding_and_monotonic_residual():
    p=toy()
    zero=torch.zeros((1,1,*TEACHER_SHAPE))
    warped0=zero.clone()
    warped1=zero.clone()
    warped0[0,0,16,16,15]=0.1
    warped1[0,0,16,16,15]=0.1
    warped0[0,0,16,17,15]=0.5
    warped1[0,0,16,17,15]=0.5
    w, report = reliability_from_image_residuals(p,warped0,warped1,zero)
    assert w[0,0,16,16,15] > w[0,0,16,17,15]
    assert w[0,0,19,19,15] == 0
    assert w[0,0,0,0,0] == 0
    assert report["per_class"]["RV"]["candidate_voxels"] > 0
    assert report["per_class"]["LV"]["candidate_voxels"] == 0
    assert report["per_class"]["LV"]["mean_image_weight"] is None


def test_invalid_input_geometry_rejected():
    p=toy()
    zeros=torch.zeros((1,1,*TEACHER_SHAPE))
    with pytest.raises(ValueError,match="geometry"):
        reliability_from_image_residuals(p,zeros,zeros[:,:,:,:,:31],zeros)
    with pytest.raises(ValueError,match="Nonfinite"):
        bad=zeros.clone()
        bad[0,0,3,3,12]=float("nan")
        reliability_from_image_residuals(p,zeros,bad,zeros)


def test_pseudo_loss_ignores_invalid_and_has_finite_gradients():
    logits = torch.zeros((1,4,2,2,2),requires_grad=True)
    pseudo = torch.full((1,1,2,2,2), 255, dtype=torch.long)
    weight = torch.zeros_like(pseudo,dtype=torch.float32)
    zero_loss=class_balanced_pseudolabel_ce(logits,pseudo,weight)
    assert zero_loss.item()==0
    zero_loss.backward(retain_graph=True)
    assert torch.isfinite(logits.grad).all()
    pseudo[0,0,0,0,0]=1
    pseudo[0,0,0,0,1]=2
    weight[0,0,0,0,0]=1
    weight[0,0,0,0,1]=1
    logits.grad=None
    ce=class_balanced_pseudolabel_ce(logits,pseudo,weight)
    assert ce.item()==pytest.approx(np.log(4),rel=1e-6)
    ce.backward()
    assert torch.isfinite(logits.grad).all()


def test_pseudo_loss_no_background_dominance():
    logits = torch.zeros((1,4,1,1,4),requires_grad=True)
    pseudo=torch.tensor([0,0,0,1],dtype=torch.long).view(1,1,1,1,4)
    weight=torch.ones_like(pseudo,dtype=torch.float32)
    # foreground error high, background error near zero
    with torch.no_grad():
        logits[0,0,0,0,:3]=8.0
        logits[0,1,0,0,3]=-8.0
    normal=class_balanced_pseudolabel_ce(logits,pseudo,weight,background_factor=1.0)
    reduced=class_balanced_pseudolabel_ce(logits,pseudo,weight,background_factor=0.25)
    assert reduced > normal


def test_ignore_confidence_error_fails_closed():
    logits=torch.zeros((1,4,1,1,1),requires_grad=True)
    label=torch.full((1,1,1,1,1),255)
    weight=torch.ones_like(label,dtype=torch.float32)
    with pytest.raises(ValueError,match="Ignored"):
        class_balanced_pseudolabel_ce(logits,label,weight)


def test_validation_patient_excluded_by_split_guard():
    assert split_role("patient090") == "train"
    assert split_role("patient091") == "validation"
    with pytest.raises(ValueError):
        split_role("patient101")
