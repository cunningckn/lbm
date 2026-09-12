import json
import resource
import tempfile
from pathlib import Path

from lbm.dataloader.mmap.frame_mmap_io import _cache_is_ready, _commit_jpeg_pack

with tempfile.TemporaryDirectory(prefix='lbm-rss-') as work:
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    root = Path(work)
    _commit_jpeg_pack(root, (b'x' * 4096 for _ in range(262144)), source_tag='stress',
                      jpeg_quality=85, image_size=None, height=8, width=8)
    assert _cache_is_ready(root, 'stress')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    delta_mib = (peak - before) / 1024
    print(json.dumps({'payload_mib': (root/'frames.bin').stat().st_size/2**20,
                      'num_frames': 262144, 'peak_rss_mib': peak/1024,
                      'additional_peak_rss_mib': delta_mib}))
    assert delta_mib < 128, delta_mib
