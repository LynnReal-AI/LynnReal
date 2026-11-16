import pytest
from lynnreal_hand.benchmark import summarize


def test_delivery_fps_includes_commit_and_housekeeping():
    def row(i,mtp,interval,**kw):
        return dict(warmup=False,chunk=i,latent_frames=5,RGB_frames=17,
                    mtp_ms=mtp,output_interval_ms=interval,commit_after_RGB_ms=200,**kw)
    records=[row(2,700,1000),row(3,800,1100),row(0,100,100),
        dict(row(2,1,1),warmup=True),dict(row(4,100,200),latent_frames=2,RGB_frames=5)]
    result=summarize(records)
    assert result['measured_chunks']==2
    assert result['MTP_mean_ms']==750 and result['MTP_median_ms']==750
    assert result['continuous_FPS']==pytest.approx(34/2.1)
    assert result['minimum_observed_ms']==700


def test_no_fake_steady_result_for_first_chunks_or_native_tail():
    with pytest.raises(ValueError,match='No formal steady'):
        summarize([dict(warmup=False,chunk=1,latent_frames=5)])
