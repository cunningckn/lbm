# MolmoMotion 未完成项实施计划

日期：2026-09-15。基线：`0f51dba`；隔离分支：`codex/molmo-motion-completion`。

## 目标与边界

保留已完成的数值 mmap 数据，不改动原工作目录、既有 v1 成品或失败现场。补齐可验证的发布/续跑契约，提供可迁往金山云的独立交付路径，逐步实现 RGB 随机读取。不得把“发布源文件齐全”“数值缓存完成”写成“全模态训练数据完成”。不修改模型、loss 或强制接入训练框架。

源：`tc_dev:/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m`。旧产物：`/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1`。新测试产物放在独立 `completion-tests` 目录，不覆盖 v1。全量新版本发布前必须通过以下验收；不自动对 339 GB 成品作破坏性重建。

## P0：发布完整性和安全续跑

- [x] 严格解析校验清单：拒绝重复、越界、符号链接、遗漏文件和非法摘要；建立 READY→清单→元数据/内容的完整校验链。
- [x] 续跑绑定源快照内容摘要、schema、代码版本、构建参数；现存旧格式必须显式迁移/验证，不能只凭 READY 跳过。
- [x] 读取端拒绝未完成或未知版本数据。构建内部校验明确走非发布入口，不能降低公共读接口约束。
- [x] 新 release 根清单覆盖组件 READY/清单及根元数据；组件完整哈希校验为可选耗时步骤，manifest-only 不宣称内容已验证。
- [x] 测试：损坏、缺失、额外文件、非法路径、参数变化、缺失 READY、版本变化均拒绝；合法旧成品只读检查仍可用。

## P1：交付与迁移

- [x] export 命令仅导出显式列入清单的正式文件，排除 `_jobs`、pilot、partial 和日志；目标存在时拒绝覆盖。
- [x] staging 写入，校验后 READY 最后发布；拷贝不依赖源硬链接或绝对运行路径。提供根清单及明确的 annotations-only 能力声明。
- [x] relocation-test 在原路径不可访问的独立进程读取迁移副本，不写缓存；先小样本，再实际金山云验收。全量网络迁移必须有可达通道和目标目录，不假定已执行。
- [x] 补 SCHEMA.md、MIGRATION.md、缺失模态/异常记录；明确坐标、mask 推导、dtype 转换与不确定字段，不能猜测单位。

## P2：图像缓存（先真实 pilot）

- [x] MolmoSpaces 已有 MP4：按视频/视角去重；从 tar offset 读取，不全量散开小文件；流式写 `frames.bin` 与 uint64 offset/length、frame index、PTS 索引。（pilot中PNG/JPEG分别一个bin）
- [ ] 默认原尺寸；PNG 无损验证基线与 JPEG q95 候选比较。帧数、顺序、时间戳与轨迹逐 clip 核对；不静默截断或重采样。每 shard 独占写，记录输入/config/schema/code 指纹与哈希，安全续跑。
- [x] reader 支持显式 frame/point 索引，返回所选 ID 和时间，RGB 缺失必须报错或显式 annotations-only。
- [x] 先抽至少不同长度/视角真实视频验证，测量峰值内存和膨胀率；据结果决定 2–8 GiB shard 和并发。未验证前不提交全量高资源任务。（四段pilot已测，建议下一阶段2GiB/4worker受控扩样；生产打包未实现）
- [x] DROID/EgoDex/HD-EPIC/YTVIS RGB、Xperience RGB/相机、Stereo4D 轨迹/相机/RGB：列明上游获取与许可依赖；未取得文件时状态为 blocked，不能伪造重建。

## P3：独立语义与吞吐验收

- [ ] 独立于 builder 的源读取比较：跨子集、split、object/hand、首尾帧、NaN/mask、相机稀疏帧索引、原始 dtype；适用时验证投影几何。
- [ ] 相同样本、窗口、点数和图像质量比较 raw/cache；记录初始化、稳态、吞吐、延迟和 RSS，分别测 0/1/多 worker。首次访问不冒称系统冷缓存，不清共享主机 page cache。
- [x] 分别报告正式交付逻辑字节、分配字节、硬链接去重增量及 raw 固定快照同口径大小；不要把失败产物算入交付。
- [x] 报告明确 annotations-only、RGB pilot 和全模态三种状态；原有 11–29x 只适用于小规模预热数值窗口，不能外推训练吞吐。

## 2026-09-15 实施状态

详见 `MOLMO_MOTION_COMPLETION_REVIEW_20260915.md` 和同目录 JSON 证据。
P0/P1实现与小样本验证通过，未重建完整新release。金山云实际完成512条样本迁移读取。
P2时间对齐未通过：15FPS标注与约15.15FPS编码时钟差异，strict-time拒绝发布；
显式frame-index实验保留双时钟并标记未验证。PNG膨胀78.86倍，JPEG q95膨胀25.36倍（四视频）。
P3六个数值子集独立源字段抽检通过；图像0/1/4worker已测。全量语义/投影几何、
实际训练框架端到端吞吐仍未认证，故相关综合项保留未勾选。

## 执行交接

先 P0 自动化测试，再 P1 小型迁移，再 P2 真视频 pilot 与 P3 配对测量。每阶段更新实际命令、结果和未解决依赖；任何失败保留独立 staging，不删除旧成品。提交采用 `fix(data): ...`、`feat(data): ...`、`docs(data): ...`，不重写他人历史。最终报告以测试输出和产物位置为准，未完成项继续保留未勾选。
