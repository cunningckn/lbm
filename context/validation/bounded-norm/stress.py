import json
import resource
import time

import numpy as np

from lbm.utils.preprocess import _Moments

block = np.tile(np.arange(65536, dtype=np.float32)[:, None], (1, 16))
started = time.monotonic()
baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with _Moments(scratch_dir='/tmp') as m:
    for _ in range(256):
        m.update(block)
    stats = m.as_dict()
    np.testing.assert_allclose(stats['q01'], 655.)
    np.testing.assert_allclose(stats['q99'], 64880.)
print(json.dumps({'payload_bytes':block.nbytes*256, 'rows':stats['count'],
                  'extra_peak_mib':(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-baseline)/1024,
                  'seconds':time.monotonic()-started}))
