# Exact quantiles with bounded reducer memory

`DiskQuantiles` stores float32 observations in an automatically cleaned-up
scratch file. Four sequential radix-selection passes locate the order
statistics for linear interpolation, preserving the previous NumPy quantile
semantics without retaining or sorting all observations in RAM. Mean/std
accumulation also uses fixed-size blocks. The dataset adapter's episode
vector memory remains separate; this change bounds the reducer's workspace.

On js_dev_2, the attached stress script processed 1 GiB of observations
(16,777,216 rows × 16 coordinates), checked known exact q01/q99 values,
and measured 25.55 MiB of additional peak RSS in 22.12 seconds.
The scratch file was removed when the computation completed.

Regression: randomized multi-block comparisons with NumPy, repeated reads
and appends, constants, signed extreme values, and failure cleanup passed.
CPU suite: 438 passed, 17 resource skips, 14 integration cases deselected.
Scratch I/O errors propagate; the atomic norm writer preserves an old
statistics file if the computation or write fails.
