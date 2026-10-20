"""Additional contract tests after the V2 review. No nibabel or HF download."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from pulse_repro import cine_data
from pulse_repro.cine_v2 import PULSECineReliabilityV2, load_cine_v2, masked_probe_ce
from pulse_repro.train_cine_v2 import run
from test_cine_v2 import MiniPulse


class FakeNii:
    def __init__(self, p):
        self.z = np.load(p)
        self.dataobj = self.z['volume']
        self.affine = self.z['affine']
        self.shape = self.dataobj.shape


def save_fake_nii(path, volume, affine=None):
    with path.open('wb') as f:
        np.savez(f, volume=volume, affine=np.eye(4) if affine is None else affine)


def prepare_patient(tmp_path, monkeypatch):
    rng = np.random.default_rng(10)
    raw = tmp_path/'raw'/'training'/'patient001'
    normalized = tmp_path/'normalized'/'training'/'patient001'
    raw.mkdir(parents=True)
    normalized.mkdir(parents=True)
    data = rng.normal(size=(12,10,4,8)).astype(np.float32)
    save_fake_nii(raw/'patient001_4d.nii', data)
    (raw/'Info.cfg').write_text('ED: 2\nES: 5\nNbFrame: 8\nGroup: NOR\n')
    for frame in (2,5):
        prefix=f'patient001_frame{frame:02d}'
        save_fake_nii(normalized/f'{prefix}.nii',data[:,:,:,frame-1])
        save_fake_nii(normalized/f'{prefix}_gt.nii',np.zeros((12,10,4),np.uint8))
    monkeypatch.setattr(cine_data,'_nib',lambda:SimpleNamespace(load=FakeNii))
    return raw,normalized.parent.parent,data


def test_phase_preflight_rejects_shifted_time_index_and_nbframe(tmp_path,monkeypatch):
    raw,norm,data=prepare_patient(tmp_path,monkeypatch)
    info=cine_data.verify_patient_cine(raw,norm)
    assert info['phase_checks']['ED']['indexed_frame_ncc']>.999
    # Bad 3D ED file is a different exact cine frame, not a valid ED target.
    save_fake_nii(norm/'training'/'patient001'/'patient001_frame02.nii',data[:,:,:,2])
    with pytest.raises(ValueError,match='frame index|BETTER'):
        cine_data.verify_patient_cine(raw,norm)
    save_fake_nii(norm/'training'/'patient001'/'patient001_frame02.nii',data[:,:,:,1])
    (raw/'Info.cfg').write_text('ED: 2\nES: 5\nNbFrame: 7\nGroup: NOR\n')
    with pytest.raises(ValueError,match='NbFrame'):
        cine_data.verify_patient_cine(raw,norm)


def test_no_temporal_ablation_requires_disabling_both_probe_losses():
    with pytest.raises(ValueError,match='disable consistency AND probe'):
        run(Path('/missing'),Path('/missing'),Path('/tmp/unused-v2'),
            use_temporal=False)


def test_checkpoint_state_dict_and_config_are_strictly_loadable(tmp_path,monkeypatch):
    torch.manual_seed(17)
    model=PULSECineReliabilityV2(MiniPulse(),use_structure=False,use_temporal=True)
    x=torch.randn(1,3,16,16); t=torch.randn(1,2,16,16)
    ph=torch.tensor([[.2,.98,.05]])
    ck=tmp_path/'model.pth'
    torch.save({'variant':'cine-reliability-v2','fold':0,
                'use_structure':False,'use_temporal':True,
                'protocol_version':'audited-directional-v2',
                'model':model.state_dict()},ck)
    import sys
    monkeypatch.setitem(sys.modules,'pulse_repro.runtime',SimpleNamespace(
        upstream=lambda p:(None,SimpleNamespace(PULSESeg=lambda s:MiniPulse()))))
    after=load_cine_v2(tmp_path,ck,torch.device('cpu'))
    with torch.no_grad():
        original=model.eval()(x,t,ph)
        loaded=after(x,t,ph)
    for a,b in zip(original,loaded):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert not after.decoder.enable_structure and after.decoder.enable_temporal


def test_max_steps_rejected_for_multiepoch_publication_run():
    with pytest.raises(ValueError,match='one-epoch smoke'):
        run(Path('/missing'),Path('/missing'),Path('/tmp/unused-v2'),
            max_steps=3,epochs=120)


def test_ablation_disables_unused_module_gradients():
    model=PULSECineReliabilityV2(MiniPulse(),use_structure=False,use_temporal=False)
    assert all(not p.requires_grad for p in model.decoder.spatial.enhancers.parameters())
    assert all(not p.requires_grad for p in model.decoder.temporal.parameters())
    assert any(p.requires_grad for p in model.base.stem.parameters())


def test_probe_cross_entropy_is_masked_on_invalid_warp():
    from torch.nn import functional as F
    logits=torch.randn(1,4,8,8,requires_grad=True)
    labels=torch.randint(0,4,(1,8,8))
    valid=torch.ones((1,1,8,8),dtype=torch.bool)
    torch.testing.assert_close(masked_probe_ce(logits,labels,valid),
                               F.cross_entropy(logits,labels),atol=1e-7,rtol=1e-7)
    valid[:,:,:,4:]=False
    first=masked_probe_ce(logits,labels,valid)
    altered=logits.clone()
    altered[:,:,:,4:]+=1000
    second=masked_probe_ce(altered,labels,valid)
    torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-6)
    none=masked_probe_ce(logits,labels,torch.zeros_like(valid))
    assert none.item() == 0
    none.backward()
    assert torch.isfinite(logits.grad).all()
