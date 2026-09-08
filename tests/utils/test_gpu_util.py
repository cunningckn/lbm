from lbm.utils.gpu_util import GpuSample, parse_nvidia_smi_csv, parse_smi_text, summarize

HY_DUMP = """
HCU[0]		: Average Graphics Package Power (W): 412.0
HCU[0]		: Average GFX Core Power (W): 280.0
HCU[0]		: HCU use (%): 88.5
HCU[0]		: HCU memory use (%): 72
HCU[0]		: sclk clock level: 3 (1600Mhz)
HCU[0]		: HCU util in last second : 91%
HCU[0]		: CU util in last second : 64%
HCU[0]		: wave util in last second : 41%
HCU[0]	: Bandwidth: | R: 102400.00 MB/s           | W: 51200.00 MB/s           | R_W: 153600.00 MB/s
"""


def test_parse_hy_smi_metrics():
    rows = parse_smi_text(HY_DUMP)
    assert 0 in rows
    sample = rows[0]
    assert sample.gpu == 91.0
    assert sample.sm == 64.0
    assert sample.wave == 41.0
    assert sample.mem == 72.0
    assert sample.power_w == 412.0
    assert sample.gfx_w == 280.0
    assert sample.sclk_mhz == 1600.0
    assert abs(sample.membw_gbps - 150.0) < 0.01


def test_hcu_use_fills_gpu_when_util_missing():
    text = "HCU[1]\t\t: HCU use (%): 55.0\n"
    sample = parse_smi_text(text)[1]
    assert sample.gpu == 55.0


def test_summarize_means():
    out = summarize(
        [
            GpuSample(gpu=10.0, sm=20.0),
            GpuSample(gpu=30.0, sm=40.0, wave=8.0),
        ]
    )
    assert out.gpu == 20.0
    assert out.sm == 30.0
    assert out.wave == 8.0


def test_parse_nvidia_csv():
    sample = parse_nvidia_smi_csv("97, 80, 350.5, 1410\n", 0)
    assert sample is not None
    assert sample.gpu == 97.0
    assert sample.mem == 80.0
    assert sample.power_w == 350.5
    assert sample.sclk_mhz == 1410.0
