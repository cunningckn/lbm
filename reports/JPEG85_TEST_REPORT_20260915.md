# MolmoSpaces JPEG q85 对照测试

2026-09-15；分支 `codex/molmo-motion-completion`。

## 结论

同一组四段视频、1064帧、352×624原尺寸，q85的JPEG帧数据比q95减少43.93%。
图像有损误差增大，但抽查的四张中心裁剪中，主体和夹爪仍可辨认；细纹理和边缘有损失。
这支持将q85作为扩大样本测试的候选，不构成训练效果无损的证明。

| 指标 | q95 | q85 |
| --- | ---: | ---: |
| JPEG二进制帧文件合计 | 57,046,658 bytes | 31,988,546 bytes |
| 相对原MP4成员容量 | 25.36× | 14.22× |
| 每视频RGB像素PSNR范围 | 41.46–44.22 dB | 38.19–41.56 dB |

上述容量只统计JPEG帧文件，不含索引和测试用PNG。
原MP4共2,249,175字节；独立JPEG帧缺少视频的帧间压缩，因此即使q85仍显著膨胀。
源内容/PNG帧二进制hash、frame_id、PTS、time_base在两组中完全一致。
两组均以decoded RGB24为参照；Pillow编码参数仅quality不同，其余沿用相同默认值。

## 同机配对读取

每视频12次请求、每次8个固定随机帧，共48次。每种worker配置交替运行q85/q95三轮，
取整段吞吐中位数，包含reader初始化和子进程启动。q95是本轮重新测量，不与历史单次结果混算。

| worker | q95请求/秒 | q85请求/秒 |
| --- | ---: | ---: |
| 0 | 74.07 | 87.51 |
| 1 | 56.42 | 64.84 |
| 4 | 137.23 | 147.67 |

q85本轮约快8%–18%。小样本且OS page cache未受控，不代表全量冷盘或训练吞吐。
两种有损质量不同，因此不宣称同画质性能提升。完整每轮结果见JPEG85_QUALITY_COMPARISON.json。

## 对比图与范围

JPEG85_COMPARISON.png每行来自一段视频的中间帧，展示中心192×192裁剪，最近邻放大2倍；
三列依次为原始解码PNG、q95、q85。四段均来自同一house、两种视角及两段动作，代表性有限。
PSNR为全部1064帧的逐视频总体MSE计算，而非仅对截图计算；未测SSIM和训练指标。

## 代码与复现

新增`--jpeg-quality`参数，默认保留95；质量参数进入构建契约，改变质量时拒绝复用旧目录。
13项已有自动化测试通过，修改文件Ruff及diff检查通过。q85产物完整文件哈希校验通过。
本次显式使用frame-index实验策略，保留原始PTS；JPEG质量变化不改变已有时间语义状态。

腾讯云测试目录：
`/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/completion-tests/rgb-q85-20260915`

```bash
cd /home/tione/workspace/kainingchen/lbm-completion-20260915/tools/molmo_motion_cache
export PYTHONPATH=src
python3 -m molmo_motion_cache.rgb_pilot \
  /home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1 \
  /new/path/rgb-q85 --alignment frame-index --jpeg-quality 85
python3 compare_jpeg_quality.py /new/path/rgb-q85 \
  /home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/completion-tests/rgb-frame-index-20260915
```

保留q95和旧v1，没有启动全量转换。下一步建议扩大house/小目标/遮挡场景覆盖后，再决定正式质量。
