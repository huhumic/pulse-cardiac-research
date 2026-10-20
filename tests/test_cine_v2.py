"""Fast CPU contract tests; no HF checkpoint or external ACDC files needed."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse_repro.cine_data import (
    build_temporal_input, find_cine, neighboring_indices, phase_context,
    _phase_similarity
)
from pulse_repro.cine_v2 import (
    BidirectionalEvidenceFusion, PULSECineReliabilityV2,
    reliable_agreement_loss, warp_2d
)


def test_temporal_index_wrap_and_one_based():
    assert neighboring_indices(1, 20) == (0,19,1)
    assert neighboring_indices(20,20) == (19,18,0)
    assert neighboring_indices(7,20) == (6,5,7)
    with pytest.raises(ValueError): neighboring_indices(0,20)
    with pytest.raises(ValueError): neighboring_indices(5,2)


def test_build_temporal_source_is_time_not_spatial():
    h,w,z,t=8,9,4,6
    cine=np.zeros((h,w,z,t),np.float32)
    for ti in range(t):
        for zi in range(z):
            yy,xx=np.mgrid[:h,:w]
            cine[...,zi,ti]=3*xx + zi*yy+ti*(xx+yy*2)**2/20
    target=cine[...,2].copy()
    spatial,temporals,phase=build_temporal_input(target,cine,1,frame=3,ed_frame=1,size=32)
    np.testing.assert_allclose(phase,phase_context(3,6,1))
    assert spatial.shape==(3,32,32)
    assert temporals.shape==(2,32,32)
    assert not np.allclose(temporals[0],temporals[1])
    assert not np.allclose(temporals[0],spatial[0])


def test_four_d_missing_is_a_hard_error(tmp_path):
    p=tmp_path/"patient001"
    p.mkdir()
    (p/"patient001_frame01.nii").write_bytes(b"placeholder")
    with pytest.raises(FileNotFoundError,match="4d"):
        find_cine(p)


def test_zero_offset_feature_warp_is_identity():
    image=torch.rand(2,5,11,13)
    grid=torch.zeros((2,2,11,13))
    got,mask=warp_2d(image,grid,return_valid=True)
    assert bool(mask.all())
    torch.testing.assert_close(got,image,rtol=2e-5,atol=2e-6)
    with pytest.raises(ValueError):warp_2d(image,grid[:,:,:-1])


def test_bidirectional_fusion_well_formed_and_zero_init():
    torch.manual_seed(7)
    block=BidirectionalEvidenceFusion(decoder_channels=8,temporal_channels=8)
    img=torch.rand(2,1,24,24)
    neighbors=torch.rand(2,2,24,24)
    phase=torch.tensor([[0.,1.,.05]]).repeat(2,1)
    delta,aux=block(img,neighbors,(8,8),phase)
    assert delta.shape==(2,8,8,8)
    assert delta.abs().max().item()==0.
    assert aux["weights"].shape==(2,3,8,8)
    assert aux["valid_minus"].shape == (2,1,8,8)
    torch.testing.assert_close(aux["weights"].sum(1),torch.ones(2,8,8),atol=1e-6,rtol=1e-6)
    assert aux["probe_minus"].shape==(2,4,8,8)
    (delta.square().mean()+aux["probe_minus"].mean()).backward()
    assert block.probe.weight.grad is not None
    assert torch.isfinite(block.probe.weight.grad).all()


def test_agreement_loss_zero_when_probes_equal_and_grad():
    logits=torch.randn(2,4,8,8,requires_grad=True)
    weights=torch.tensor([.1,.45,.45]).reshape(1,3,1,1).expand(2,-1,8,8)
    info={"probe_minus":logits,"probe_plus":logits.clone(),"weights":weights}
    target=torch.randn(2,4,16,16)
    zero=reliable_agreement_loss(info,target,threshold=0.)
    assert abs(zero.item())<1e-7
    other={"probe_minus":logits,"probe_plus":torch.randn(2,4,8,8),"weights":weights}
    l=reliable_agreement_loss(other,target,threshold=0.)
    assert l.item()>0
    l.backward()
    assert torch.isfinite(logits.grad).all()


class MiniDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_size=64
        self.projs=nn.ModuleList(nn.Sequential(nn.Linear(8,8),nn.GELU()) for _ in range(4))
        self.res=nn.ModuleList(nn.Identity() for _ in range(4))
        self.ups=nn.ModuleList(nn.Upsample(scale_factor=2,mode="nearest") for _ in range(4))
        self.head=nn.Conv2d(8,4,1)
        self.aux1=nn.Conv2d(8,4,1)
        self.aux2=nn.Conv2d(8,4,1)

    def _reassemble(self,h,proj):
        t=h[:,1:];b,n,c=t.shape
        s=math.isqrt(n)
        return proj(t).transpose(1,2).reshape(b,-1,s,s)

    def forward(self,hs):
        f=[self._reassemble(h,self.projs[i]) for i,h in enumerate(hs)]
        x=self.ups[3](self.res[3](f[3]))
        x=self.ups[2](self.res[2](x+F.interpolate(f[2],size=x.shape[-2:],mode="bilinear",align_corners=False)))
        a1=self.aux1(x)
        x=self.ups[1](self.res[1](x+F.interpolate(f[1],size=x.shape[-2:],mode="bilinear",align_corners=False)))
        a2=self.aux2(x)
        x=self.ups[0](self.res[0](x+F.interpolate(f[0],size=x.shape[-2:],mode="bilinear",align_corners=False)))
        return (F.interpolate(self.head(x),(64,64),mode="bilinear",align_corners=False),a1,a2)


class MiniViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Linear(3,8)
        self.encoder=nn.Module()
        self.encoder.layer=nn.ModuleList([nn.Linear(8,8) for _ in range(12)])

    def forward(self,pixel_values,output_hidden_states=True):
        x=F.adaptive_avg_pool2d(pixel_values,(4,4)).flatten(2).transpose(1,2)
        x=self.embed(x)
        s=torch.zeros((x.shape[0],1,8),device=x.device)
        hs=[torch.cat((s,x),1)]
        for block in self.encoder.layer:
            x=block(x)*.01+x
            hs.append(torch.cat((s,x),1))
        class R:pass
        r=R();r.hidden_states=hs
        return r


class MiniPulse(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv2d(3,3,1)
        self.vit=MiniViT()
        self.backbone_blocks=self.vit.encoder.layer
        self.decoder=MiniDecoder()

    def forward(self,x):
        out=self.vit(pixel_values=self.stem(x),output_hidden_states=True)
        return self.decoder([out.hidden_states[i] for i in (3,6,9,12)])


def test_v2_initial_logits_near_original_pulse_and_backward():
    torch.manual_seed(11)
    src=MiniPulse()
    ref=copy.deepcopy(src).eval()
    model=PULSECineReliabilityV2(src,use_structure=True).eval()
    x=torch.rand(2,3,16,16)
    t=torch.rand(2,2,16,16)
    phase=torch.tensor([[.2,.98,.05]],dtype=torch.float32).repeat(2,1)
    with torch.no_grad():
        expected=ref(x)
        actual=model(x,t,phase)
    for a,b in zip(expected,actual):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    model.train()
    scores=model(x,t,phase,return_aux=True)
    loss=F.cross_entropy(scores[0],torch.randint(0,4,(2,64,64)))
    loss=loss+.1*reliable_agreement_loss(scores[3],scores[0],threshold=0.)
    loss.backward()
    assert model.decoder.temporal.to_decoder.weight.grad is not None
    assert torch.isfinite(model.decoder.temporal.to_decoder.weight.grad).all()
    assert model.base.stem.weight.grad is not None
    assert torch.isfinite(model.base.stem.weight.grad).all()


def test_temporal_shape_violation():
    model=PULSECineReliabilityV2(MiniPulse())
    with pytest.raises(ValueError,match="B2HW"):
        model(torch.randn(1,3,16,16),torch.randn(1,3,16,16),torch.tensor([[0.,1.,.05]]))


def test_temporal_swap_changes_direction_conditioned_evidence_after_training():
    torch.manual_seed(9)
    block = BidirectionalEvidenceFusion(decoder_channels=8,temporal_channels=8)
    with torch.no_grad():
        block.to_decoder.weight.normal_(0,.05)
        # Controlled non-zero direction signal: proves the architecture can
        # distinguish chronological order, not that an untrained network does.
        block.direction_embed[0].fill_(.5)
        block.direction_embed[1].fill_(-.5)
    target=torch.rand(1,1,32,32)
    neighbors=torch.rand(1,2,32,32)
    phase=torch.tensor([[.5,.866,.05]])
    a,_=block(target,neighbors,(16,16),phase)
    b,_=block(target,neighbors.flip(1),(16,16),phase)
    assert (a-b).abs().max().item()>1e-6


def test_invalid_warp_locations_are_explicitly_rejected():
    image=torch.rand(1,3,9,9)
    offsets=torch.zeros(1,2,9,9)
    offsets[:,0]+=5.0
    shifted,valid=warp_2d(image,offsets,return_valid=True)
    assert (~valid).any() and valid.any()
    assert (shifted*~valid).abs().max()==0


def test_true_phase_correlation_and_wrong_time_frame():
    rng=np.random.default_rng(6)
    image=rng.normal(size=(12,10,4)).astype(np.float32)
    assert _phase_similarity(image,2*image+12)>0.9999
    other=rng.normal(size=(12,10,4)).astype(np.float32)
    assert _phase_similarity(image,other)<.8


def test_no_temporal_ablation_is_independent_of_neighbors():
    torch.manual_seed(5)
    model=PULSECineReliabilityV2(MiniPulse(),use_temporal=False)
    x=torch.randn(1,3,16,16)
    phase=torch.tensor([[.0,1.,.05]])
    with torch.no_grad():
        one=model.eval()(x,torch.zeros(1,2,16,16),phase)[0]
        two=model.eval()(x,torch.rand(1,2,16,16),phase)[0]
    torch.testing.assert_close(one,two,atol=0,rtol=0)


def test_direction_is_required_and_invalid_phase_rejected():
    blk=BidirectionalEvidenceFusion(8,8)
    with pytest.raises(ValueError,match="phase_context"):
        blk(torch.rand(1,1,16,16),torch.rand(1,2,16,16),(8,8),torch.zeros(1,2))
    with pytest.raises(ValueError,match="Invalid"):
        blk(torch.rand(1,1,16,16),torch.rand(1,2,16,16),(8,8),torch.zeros(1,3))


def test_cine_phase_is_ed_anchored_not_file_start():
    np.testing.assert_allclose(phase_context(6,20,6),
        [0.,1.,.05],atol=1e-6)
    np.testing.assert_allclose(phase_context(7,20,7),
        [0.,1.,.05],atol=1e-6)
    np.testing.assert_allclose(phase_context(9,20,6),
        phase_context(11,20,8),atol=1e-6)
