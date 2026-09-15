# MolmoMotion 转存重构验证记录

日期：2026-09-16。基线：`origin/feature/tc_covert_adapt` 的 `0f51dba`。实施分支：`codex/molmo-refactor-20260916`。本次只改转换工具与测试；没有覆盖原始数据 `/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m`，也没有覆盖既有成品 `/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1`。

## 已交付的重构

- `generic.py` 不再承担每个子集的源格式解析。EgoDex、HD-EPIC、MolmoSpaces、Xperience、YTVIS 的 tar/NPZ/JSON 解析放入 `adapters/`，写分片、来源校验和输出元数据仍由通用构建器负责。
- `generic_reader.py` 只依赖成品的 Parquet/NPY/完成标记，不再导入构建器或访问原始 tar。小包复制后屏蔽原始数据路径的读取测试已通过。
- DROID、通用数值子集、Stereo4D 元数据和 portable assets 共用完成标记/清单校验逻辑。Stereo4D 明确标为 metadata-only，不能误认为含轨迹、相机或 RGB。
- 完整 release 新增 Hugging Face tree revision、snapshot fingerprint 和 component fingerprint。MolmoSpaces 数值与 assets 配对先校验这些身份，不能仅凭相同 video ID、文件名或大小混用不同来源版本。
- Xperience 的 reader 和 benchmark 同时保留轨迹、`trust_weights`、`clip_frame_indices` 与 `source_point_indices`。新产物显式存点身份；旧产物从 `keep_mask` 推导，避免丢失原始点对应关系。
- MolmoSpaces 姿态约定统一为：`points3d` 在世界坐标系，`camera_poses` 为 camera-to-world；投影时使用其逆矩阵。转换不再隐式翻转矩阵方向。
- 单 worker 的 tar 索引、数值写分片和源哈希检查在当前进程串行完成；多 worker 仍使用进程池。这样小样本不会在 Python 3.12 中无谓 fork。

## 第四步：图像读取对齐 LBM

新增独立模块 `molmo_motion_cache.rgb224`，没有导入 LBM 训练包。它面向既有的 JPEG 分片格式：`frames.npy` 记录 `(shard, offset, length)`，载荷为 `frames-00000.bin` 等文件。

- `Rgb224FrameReader` 将每个 bin 分片映射为只读 mmap，并以 `max_open_shards` 做 LRU 上限。
- `borrow_jpeg()` 只在上下文内暴露 `memoryview`；退出时先释放视图，再允许关闭或淘汰映射。活动视图存在时 `close()` 会明确报错。
- fork 或 spawn 后 reader 会在子 worker 中重新打开 index/mmap，不复用父 worker 的映射和锁。
- `read_jpeg_bytes()` 与 `decode_rgb(..., mode="seek")` 保留 seek/read 回退；mmap 路径使用相同的 Pillow 解码器。Pillow 的 `BytesIO` 包装仍会复制 JPEG 字节，因此没有宣称零拷贝解码。
- 提供 `verify-rgb224` 和 `benchmark-rgb224` 命令。benchmark 会先验证相同 JPEG bytes 和相同 Pillow RGB 数组，再报告启动耗时、吞吐、p50/p95 延迟。

现有 `v1` 只有 MP4/H5 tar assets，没有 `frames.npy` 或 `frames-*.bin`，所以不能对它虚构 RGB 吞吐结果。新 reader 已用合成跨分片 payload 验证；实际 RGB 物化后必须在目标格式上执行 `verify-rgb224`、`benchmark-rgb224`，再接入真实 DataLoader 的 0/1/4 worker 测试。

## 验证证据

在 `tc_dev` 的隔离测试副本中执行：

```bash
PYTHONPATH=tools/molmo_motion_cache/src \
  /home/tione/workspace/kainingchen/xprobot_flow/.venv/bin/python \
  -m compileall -q tools/molmo_motion_cache/src tools/molmo_motion_cache/tests
PYTHONPATH=tools/molmo_motion_cache/src \
  /home/tione/workspace/kainingchen/xprobot_flow/.venv/bin/python \
  -m pytest -q tools/molmo_motion_cache/tests
```

结果：`15 passed in 0.71s`。覆盖 source adapter、完成标记、Stereo4D metadata-only、来源指纹混用拒绝、旧成品读取、Xperience 选帧选点和权重、MolmoSpaces 投影、RGB 跨分片/乱序/重复帧、LRU、活动视图、pickle 重新加载（spawn 的前置行为）和两个 RGB CLI 入口。

仓库级 pytest 收集未在 `tc_dev` 运行：既有 `tests/conftest.py` 导入的 Torch 因环境缺少 `libgalaxyhip.so.5` 失败，发生在任何 MolmoMotion 测试前。转存工具测试不导入 Torch，因此以上 15 项可独立运行；GitHub CI 的 CPU Torch 环境仍会通过根目录 `testpaths` 发现这些工具测试。

真实数据的只读检查结果：

- 旧 `v1` 的 Xperience reader 能返回从 `keep_mask` 推导的 `source_point_indices`，并保留 `trust_weights`；普通 release 清单检查通过。
- 旧 `v1` 的 source identity 状态是 `legacy-unverified`。`verify-release --require-source-identity` 与 MolmoSpaces 数值/assets 严格配对都会故意报错，要求重建到新目录，避免伪造来源身份。
- 对真实 MolmoSpaces wrist-camera 样本，camera-to-world 取逆后的 3D→2D 投影误差中位数约 `1.35e-05 px`、p95 约 `4.27e-05 px`；直接把 camera-to-world 当 world-to-camera 的中位误差约 `131.44 px`。这确认了保存约定和测试方向。

## 性能记录与边界

本次没有重新全量转存，也没有运行训练。数值数组布局、float32 精度、JPEG 编码方式和坐标变换没有改变。对现有 `v1/subsets/xperience` 做过一次只读、warm-cache、小样本检查：1 个 source record/track kind、8 次请求、4 帧、8 点、1 worker。raw tar/NPZ 为约 `405.10 samples/s`，mmap NPY/Parquet 为约 `72.97 samples/s`，校验和相同，报告比值 `0.18x`。

该测试样本太小且源 tar 已被页缓存，不能说明训练或本次重构整体退步；但它明确表明不能预设 mmap 一定更快。后续应使用固定请求集，在相同机器上分别测启动时间、稳定吞吐、p50/p95、RSS/句柄数，并覆盖 0/1/4 个真实 DataLoader worker。RGB 的同类对照由 `benchmark-rgb224` 在 RGB payload 出现后执行。

## 使用与后续动作

新 release 应使用：

```bash
molmo-motion-cache verify-release \
  --output /path/to/new-release --verify-files --require-source-identity
```

历史 `v1` 只可使用不带 `--require-source-identity` 的清单/文件校验和旧 reader；不要手工补写指纹或把新 assets 混入其中。RGB payload 物化后使用：

```bash
molmo-motion-cache verify-rgb224 --payload-root /path/to/rgb224
molmo-motion-cache benchmark-rgb224 \
  --payload-root /path/to/rgb224 --samples 256 --warmup 16 --max-open-shards 8
```

正式迁移到金山云前，复制一个小型完整组件并在目标侧执行 release/RGB 校验、随机读取和真实 DataLoader 测试。全量重转、RGB 物化、跨云传输和训练均不在本次自动执行范围内。
