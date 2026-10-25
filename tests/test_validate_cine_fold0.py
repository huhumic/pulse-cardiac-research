"""CPU-only checks for V1.5/V2 full-volume validation output."""
import csv
import pytest
from pulse_repro import validate_cine_fold0 as v

def write_csv(path, n=20, omit=None, increment=0.0):
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.writer(f); w.writerow(["patient","phase","RV_dice","Myo_dice","LV_dice"])
        for p in range(n):
            for phase in ("ED","ES"):
                if (p,phase)==omit: continue
                w.writerow([f"patient{p:03d}",phase,.8+p/1000+increment,
                            .77+p/1000+increment,.9+p/1000+increment])

def test_full_fold_and_pairing(tmp_path):
    a,b=tmp_path/"a.csv",tmp_path/"b.csv"
    write_csv(a);write_csv(b,increment=.01)
    x,y=v.load_csv(a),v.load_csv(b)
    assert v.summary_stats(x)["n_volumes"]==40
    assert v.summary_stats(x)["mean_dice"]==pytest.approx((.8095+.7795+.9095)/3)
    delta=v.paired_delta(x,y,n_bootstrap=100)
    assert delta["delta_candidate_minus_reference"]["Mean"]["mean"]==pytest.approx(.01)
    assert v.paired_delta(x,y,n_bootstrap=100)==delta

def test_missing_phase_rejected(tmp_path):
    f=tmp_path/"bad.csv";write_csv(f,omit=(19,"ES"))
    with pytest.raises(ValueError,match="exactly ED and ES"):v.load_csv(f)

def test_smoke_not_full_fold(tmp_path):
    f=tmp_path/"short.csv";write_csv(f,n=1)
    assert len(v.load_csv(f,full_fold=False))==2
    with pytest.raises(ValueError,match="Incomplete Fold 0"):v.load_csv(f)

def test_nonmatching_patient_ids_rejected(tmp_path):
    a,b=tmp_path/"a.csv",tmp_path/"b.csv";write_csv(a);write_csv(b)
    b.write_text(b.read_text().replace("patient000","patient222"))
    with pytest.raises(ValueError,match="Cannot pair cohorts"):
        v.paired_delta(v.load_csv(a),v.load_csv(b))

def test_nan_or_invalid_dice_rejected(tmp_path):
    f=tmp_path/"bad.csv";write_csv(f)
    f.write_text(f.read_text().replace(",0.8,",",1.2,",1))
    with pytest.raises(ValueError,match="Invalid RV"):v.load_csv(f)

def test_incomplete_resume_fails(tmp_path):
    d=tmp_path/"run";d.mkdir()
    with pytest.raises(FileNotFoundError):
        v.check_run(d,variant="V2",control="correct",
                    checkpoint_info={"epoch":91,"checkpoint_sha256":"abc"},full_fold=True)

def test_dry_run_does_not_create_output(tmp_path):
    assert v.run_one(acdc_root=tmp_path,pulse_root=tmp_path,
        weights=tmp_path/"fake.pth",output=tmp_path,name="V2_correct",
        control="correct",metadata={"variant":"V2","epoch":91},
        smoke=True,dry_run=True,resume=False) is None
    assert not (tmp_path/"smoke").exists()
