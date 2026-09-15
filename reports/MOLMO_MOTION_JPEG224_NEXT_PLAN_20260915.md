# MolmoMotion JPEG224 下一步实施计划

日期：2026-09-15。本次仅更新计划，交其他 agent 实施。检查的 LBM 分支为 `codex/molmo-motion-completion`，HEAD `56541c9`；开始实施时重新检查工作区，不覆盖其他 agent 改动。本计划替代旧计划中的“默认原尺寸”要求，保留历史实验报告。

## 1. 已完成与待完成

已完成且应复用：发布清单/READY/版本校验、源与代码及参数指纹续跑、组件导出、小样本跨云独立读取；六个数值子集的独立源字段抽检；四段真实视频的原尺寸 PNG/JPEG q95/q85 试验及 0/1/4 worker 测量。证据见 `MOLMO_MOTION_COMPLETION_REVIEW_20260915.md`、`JPEG85_TEST_REPORT_20260915.md`、`COMPLETION_TEST_RESULTS.json`、`MIGRATION_ACCEPTANCE.json`。已记录 13 项自动测试通过和 512 条 DROID 样本跨云验收，不能等同于完整新 release 验收。

尚未完成：JPEG224 转存；与图像一致的二维轨迹/内参变换；生产级多视频图像分片；224 尺寸下的质量、容量和端到端读取基准；完整图像发布及全量跨云迁移。既有联合读取仍默认 PNG；`get_selection` 未返回对应变换后的相机信息，必须一起补齐。

范围：先处理已有 MP4 的 MolmoSpaces；其他子集的上游 RGB，以及 Xperience 相机、Stereo4D 数值轨迹/相机仍按缺失依赖管理，不标记为全模态完成。

## 2. 必须对齐的参考函数

参考服务器：`tc_dev`。
参考文件：`/home/tione/workspace/kainingchen/mindon_pi0_dev/data/utils/image_preprocess.py`，第 77–85 行 `resize_with_pad`。
参考工作区 HEAD：`9d75617c630efc71aa7dddebc81987e3ae8689b0`。
读取时文件 SHA-256：`85c66cd60814b8169c85a2fae52026cfd80f89942fd8bb1ed3b0b08f594a5396`。
实测环境：Pillow `12.2.0`，NumPy `1.26.4`。

该函数由 `data/dataset.py` 的 `apply_image_preprocess` 路径使用；默认操作是 `resize_with_pad`。对齐的是数据预处理函数，不是 `models/vla/pi0/modeling_pi0.py:151` 或 `models/vla/pi05/modeling_pi05.py:146` 的同名 tensor 函数。后两者使用 PyTorch 插值且尺寸向下取偶数，不能视为像素等价。

固定空间变换规则：

1. 输入为解码后的 RGB uint8 HWC，不交换 RGB/BGR，不改变时间采样。
2. `image_size=224`，`pad_value=0`。
3. 使用 `ImageOps.contain(image, (224, 224), method=Image.Resampling.BILINEAR)` 等比缩放，不手写一个近似的取整公式替代。
4. 创建 RGB 黑色 224×224 画布，以 `left=(224-fitted_width)//2`、`top=(224-fitted_height)//2` 居中粘贴。余下奇数像素放右侧/底部。
5. 已是 224×224 的图像直接返回。横图补上下，竖图补左右，正方形无须补边。
6. 上述变换完成后才编码 JPEG。JPEG 有损，编码前像素一致性与编码后画质评价分开验收。

参考函数已在上述环境进行只读、内存内调用，以下尺寸规则通过；这不是新转换代码已经实现的证明。表格均为宽×高，padding 顺序为左/上/右/下。

| 输入 | 缩放后内容 | padding | 最终尺寸 |
| --- | --- | --- | --- |
| 624×352 | 224×126 | 0/49/0/49 | 224×224 |
| 854×480 | 224×126 | 0/49/0/49 | 224×224 |
| 640×480 | 224×168 | 0/28/0/28 | 224×224 |
| 480×640 | 168×224 | 28/0/28/0 | 224×224 |
| 512×512 | 224×224 | 0/0/0/0 | 224×224 |
| 641×479 | 224×167 | 0/28/0/29 | 224×224 |

不得以 LBM 现有 OpenCV `INTER_LINEAR` 版本替换后直接宣称像素对齐；相同输出尺寸并不保证与 Pillow 滤波结果相同。可以在独立工具内移植参考函数必要逻辑，保留来源/版本/哈希与对照测试；运行时不能依赖 PI0 绝对路径或导入整个训练框架。新依赖与编码参数须锁定并写入构建指纹。

## 3. 实施顺序与验收

### A. JPEG224 几何一致性

- [x] 新增独立空间变换 helper，返回 RGB224 以及源尺寸、内容尺寸、padding、变换矩阵；保留旧数值缓存不变。
- [x] 用冻结的参考函数对照横/竖/方/奇数尺寸、彩色梯度、棋盘格与真实首中尾帧。编码前逐像素相等，输出一律 `(224,224,3)`，黑边为 0；输入与元数据不一致时拒绝静默处理。
- [x] 合成例子必须包含 641×479，防止偶数截断版与 Pillow 版被混用。JPEG 编码后不要求与原始 RGB 哈希相等；黑边交界处的有损误差也不当成几何错误。
- [x] 更新 schema、构建指纹和文档：包含目标尺寸、contain/Pillow/双线性/居中规则、JPEG 质量与色度采样参数。原尺寸缓存不得冒充 224 缓存续跑。

### B. 二维轨迹、相机和时间

- [ ] 原始 points2d、points3d、内外参保持可追溯且不原地改写；联合 reader 返回明确命名的 224 坐标及内参，并返回所选 frame_id/point_id。
- [ ] 使用实际整数内容尺寸计算 `sx=fitted_width/source_width`、`sy=fitted_height/source_height`，保留实际 padding。先确认源二维坐标是像素中心还是边界约定，再固定变换矩阵 A；不能混用取整前的单一比例。
- [ ] 若坐标采用整数像素中心约定，则验证 `x'=(x+0.5)*sx-0.5+left`、`y'=(y+0.5)*sy-0.5+top` 与所用重采样规则一致；若源约定不同，先转换约定。像素域内参使用同一个 A 做 `K224=A @ Ksource`，不得对归一化内参直接套用像素公式。
- [ ] 3D 轨迹和外参不因图像 resize 改变；NaN/可见性保持原语义。另提供内容区域/有效像素 mask，不能把补边或未知坐标视为有效监督。
- [ ] 稀疏相机数据按源 frame indices 对齐，不因选择窗口就直接按数组位置索引；缺失模态显式报错或声明不可用。图像原始尺寸与标注坐标系不一致的子集需有已验证的前置映射。
- [ ] 15 FPS 标注与约 15.15 FPS 编码时间偏差单独处理：保留 PTS/time_base 与 annotation time。当前只允许显式 frame-index 实验，不通过放宽时间阈值把未验证状态改成通过；空间对齐开发可继续，但正式同步发布需独立时间语义证据。

### C. 224 真实扩样与参数选择

- [x] 先复用已有四段视频，再按固定种子跨 house、动作、视角、长度扩样，首轮至少 100 段或说明可用范围不足；输出固定样本列表。
- [ ] q85 为优先测试候选、q95 为对照，在相同 RGB224 输入上比较编码容量、有效内容区域 PSNR/SSIM、小目标和夹爪细节。不能让大片黑边抬高质量指标；既有原尺寸 q85/q95 数字只作历史对照。
- [ ] PNG 仅保留有限测试参考，正式输出只存选定质量的 JPEG，不同时全量写 PNG/q85/q95。保留原始 MP4 作为恢复源，不据此删除任何原数据。
- [ ] 统计每帧 JPEG 字节分布、全量已知帧数、逻辑/分配字节、转换速率和峰值 RSS，据此估算完整输出容量与耗时。不得把图像面积缩小比例直接当 JPEG 压缩比例。

### D. 生产图像分片与联合读取

- [ ] 基于 tar offset 流式解码，无全量散帧；按唯一视频/视角存一次图像，不按 object/hand 重复写。
- [ ] 扩样后实现约 2 GiB 目标分片，单 shard 独占写、内存有上限；视频可跨 shard 时通过全局 frame 索引解析。起始 4 worker 仅为受控实验配置，先确认资源和内存，不能自动扩大高成本作业。
- [ ] `frames.bin` + 64 位 offset/length 索引，保存 frame_id、PTS/time_base、video_id、几何元数据；完整 shard 哈希及指纹匹配后才允许恢复，未完成 staging 不发布。
- [ ] 联合 reader 显式默认 JPEG，并复用每 worker 的 reader/句柄/索引，避免每次 selection 重建 reader 或重做全清单审计；完整内容审计保留在发布/迁移阶段。输出 RGB、224 轨迹、相机数据及能力/时间验证状态。

### E. 性能与金山云交付

- [ ] 固定同样的 video/frame/point 请求、窗口和 224 空间变换。分别报告 raw MP4 在线解码+同一 resize 与 JPEG224 缓存解码；JPEG 有损差异单列，不宣称逐像素同画质加速。增加无损 RGB224 正确性基线。
- [ ] 0/1/4 worker、持久 reader、至少 3 轮交替测试，分开初始化与稳态、p50/p95、RSS 和端到端 DataLoader 吞吐。训练链路若尚未接通，明确标为独立 reader benchmark。不得清共享 page cache 或宣称未测的 GPU 利用率提升。
- [x] 先在独立新目录导出含 JPEG224、数值、空间/时间元数据的小包到金山云，执行完整哈希和原路径不可访问的联合读取验收；此前 DROID 数值迁移不替代本项。
- [ ] 以上门槛通过后才发布 MolmoSpaces 的完整新包，根清单包含图像/几何元数据和实际能力声明。完整七子集全模态仍受上游缺失数据约束。

## 4. 路径、安全与交付

源数据：`tc_dev:/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m`。
已有成品：`tc_dev:/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1`，禁止覆盖。
新实验建议放 `tc_dev:/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/completion-tests/rgb224-pi0-<run-id>`，目标存在则拒绝覆盖或严格校验指纹后恢复；正式发布用独立版本目录，验收通过后再确定切换，不静默替换 v1。
金山云小包使用既有测试根 `/mnt/kpfs/workspace/jinaoqun/Projects/lbm-completion-tests-20260915` 下新的 `rgb224-<run-id>` 子目录；全量迁移目标和可用通道必须先核实。运行日志/报告放成品外，READY 最后发布，不删除旧输出或失败现场。

交付必须包含：参考函数一致性测试、JPEG224 转存与联合 reader、跨尺寸/坐标/相机/时序验证报告、224 新画质/容量/吞吐报告、含 RGB 的跨云验收证据，以及仍受阻的模态/时间语义清单。阶段清单只有实测通过后才能勾选；本计划中的所有实施项当前均未据此宣称完成。

## 5. 实施状态更新（2026-09-15，覆盖上文初版的未实施说明）

已交付结果见 `JPEG224_IMPLEMENTATION_REPORT_20260915.md`；机器证据见同目录 `JPEG224_*.json`。19 项测试通过；100 段/26,695 帧完成 JPEG224 实验；两云四视频联合读取所有数组哈希一致。上方只勾选完整实测通过的条目，其余复合条目保留未完成，即使其中已有部分实现。

- A 完成像素/空间 helper 与参考对照。
- B 联合 reader、源/224 字段、稀疏相机与 mask 已实现；源像素中心约定仍未确认。以显式 integer-center 假设测试，状态保持 false。时间语义仍未认证，不通过放宽阈值发布。
- C 100 段扩样、q85/q95 内容区 PSNR/SSIM、容量/速率/RSS 已测；只存 q85，PNG 仅12张测试参考。ROI/训练效果尚未验证，质量尚未定为生产默认。
- D 2 GiB 目标 writer 分组、跨 shard 索引、持久 reader/句柄、完成品全哈希续跑拒绝测试已实现；自动恢复未完成组及全量长跑内存/句柄稳定性未完成。
- E 0/1/4 worker 三轮独立联合 reader benchmark 完成，发现全量数值索引初始化开销明显；不是实际训练 DataLoader 吞吐。含 RGB 跨云验收完成；正式完整包与全量迁移没有执行。

下一步先补坐标/时钟生成端证据、ROI/训练验证与全量运行门槛，不将实验包切换到 v1。
