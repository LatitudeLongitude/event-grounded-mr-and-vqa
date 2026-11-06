<p align="right"><a href="./README.zh.md"><kbd>简体中文</kbd></a> · <a href="./README.en.md"><kbd>English</kbd></a></p>

# 基于事件证据的多模态检索与视频问答

本项目是论文 **“An Evidence Framework for Event-Grounded Multimodal Retrieval and Video Question Answering”** 的配套代码，用于参加 [2nd CASTLE Multimodal Analytics Challenge 的 Fully-Automatic Track 赛道](https://castle-dataset.github.io/challenges/mm26/#gear-fully-automatic-track)。

## 项目简介

本项目面向 [CASTLE2024](https://huggingface.co/datasets/CASTLE-Dataset/CASTLE2024) 长时间、多视角视频数据，完成事件实例检索（EIS）、物体实例检索（OIS）和视频问答（QA）。系统将视频帧与数据集自带的语音转写文本组织成可追溯的证据，保留来源视角和时间信息，用于定位事件、物体及回答问题。

整体流程为：下载与抽帧 → 转写清洗与切块 → 三路特征编码 → L0–L5 分层事件构建 → 检索、重排与问答。视觉表示使用 TIPSv2，文本表示使用 Qwen3 稠密向量和 BGE-M3 稀疏向量。查询时通过三路召回、视觉局部特征重排和文本重排筛选证据，再借助事件层级确定时间区间或组织问答上下文；QA 将选出的文本、图像证据交给 OpenRouter 上的多模态模型，并验证答案及其证据引用。事件层级用于证据关联和时间定位。

| 目录 | 功能 |
|---|---|
| [download_pipeline/code](download_pipeline/code/) | 下载、校验、抽帧，以及转写清洗和文本切块。 |
| [embedding_pipeline](embedding_pipeline/) | 分别生成 TIPSv2、Qwen3 和 BGE-M3 特征。 |
| [event_pipeline](event_pipeline/) | 从帧与文本原子构建微事件、单源事件、跨源事件、活动片段和长会话。 |
| [solution](solution/) | 完成 36 道比赛题目的检索、重排、问答和提交校验，输出 EIS、OIS、QA 三份 CSV 及打包好的 ZIP。 |

## 用较小存储空间处理庞大数据集

[download_pipeline/code](download_pipeline/code/) 提供了“边下载边抽帧，抽完帧就删除”的流水线：以单个视频为处理单位，让下载、抽帧和缩放阶段交错运行，抽帧产物通过验证后便删除对应的原始视频，保留后续检索和问答需要的帧图片及来源记录，无需同时保存全部原视频。

借助这一方式，我们仅用约 **600 GB 存储空间**就完成了这个庞大数据集的下载与抽帧处理。当前配置按 1 FPS 抽帧，并将图片缩放至 1920 像素宽，同时降低图片画质。希望这一思路能启发希望使用 CASTLE数据集却没有足够存储空间保存完整原视频的研究者。

## 真实生效的参数位置

以下路径均相对于本目录。项目的有效参数分布在 YAML、命令行和 Python 源码中，不能只凭配置字段名称判断其是否参与计算。

### 数据准备与事件构建

| 环节 | 参数位置与生效范围 |
|---|---|
| 下载与抽帧 | [download_pipeline/code/config.yaml](download_pipeline/code/config.yaml) 递归覆盖 [castle_pipeline/config.py](download_pipeline/code/castle_pipeline/config.py) 的默认值，控制数据集、路径、FFmpeg、抽帧率、图片尺寸、JPEG 质量、并发、代理及 `delete_raw_video_after_verified`。`run --no-resize-frames` 可临时关闭缩放。 |
| 转写清洗 | [transcript_cleaning/config.py](download_pipeline/code/transcript_cleaning/transcript_cleaning/config.py) 定义时间戳时长的质量标记阈值，默认 1–120 秒；输入、输出路径由该模块 CLI 指定，不读取下载器 YAML。 |
| 文本切块 | [embed_chunking/config.py](download_pipeline/code/embed_chunking/embed_chunking/config.py) 定义 token 和时长阈值。CLI 可覆盖 `target_min_token_estimate`、`target_max_token_estimate`、`hard_max_token_estimate`，默认分别为 150、400、800；目标时长为 30–90 秒，硬上限为 180 秒。 |
| TIPSv2 图像编码 | [tipsv2_image_cls/code/config.yaml](embedding_pipeline/tipsv2_image_cls/code/config.yaml) 控制模型路径 `paths.model_root`、图像尺寸（448）、精度（FP32）、每卡批量（24）、GPU、解码与输出参数。`--gpu-ids`、`--decode-workers` 可覆盖对应配置；没有专用模型路径环境变量覆盖。 |
| Qwen3 文本编码 | [qwen3_text_dense/code/config.yaml](embedding_pipeline/qwen3_text_dense/code/config.yaml) 控制模型路径、最大长度（32768）、精度（float16）、批量（2）及运行参数。仅当 `model_root: auto` 时，优先读取 `QWEN3_MODEL_ROOT`，该环境变量未设置时回退到平台默认路径。 |
| BGE-M3 文本编码 | [bgem3_text_sparse/code/config.yaml](embedding_pipeline/bgem3_text_sparse/code/config.yaml) 控制模型路径、最大长度（8192）、`use_fp16`、批量（4）及运行参数。仅当 `model_root: auto` 时，优先读取 `BGEM3_MODEL_ROOT`，该环境变量未设置时回退到平台默认路径。两套文本编码的 GPU 均可由 `--gpu-ids` 覆盖，普通运行的批量和编码参数仍取自 YAML。 |
| 事件规模与视角信息 | [event_pipeline/code/config.yaml](event_pipeline/code/config.yaml) 控制输入输出路径、读取批量、`max_event_frames`、`sparse_top_n`、L2 最短时长和 L2–L5 最大跨度。[view_meta.yaml](event_pipeline/config/view_meta.yaml) 中实际生效的是视角所属角色、`location_hint` 和 `fusion_defaults.cross_source_time_slack_sec.l3_global`。 |
| 事件算法 | [event_pipeline/code/event_pipeline](event_pipeline/code/event_pipeline/) 中，`visual_features.py`、`text_features.py`、`sparse_features.py` 和 `boundary_projection.py` 控制边界特征与投影；`fusion.py` 的实际计算处定义融合权重；`segments_l1_l2.py`、`group_l3.py`、`aggregate_l4_l5.py` 控制各层构建；`representative_frames.py` 控制代表帧选择。相应权重和判定阈值需在源码中调整。 |

切块的 `soft_max_*`、`overlap_*` 不参与当前切块计算；Qwen3/BGE-M3 的 `runtime.max_retries` 未驱动重试循环，BGE-M3 的 `runtime.records_per_shard` 也不决定实际分片。事件配置的 `rows_per_parquet_group`、视角配置中的权重提示和 L1/L2/L4/L5 时间松弛量均不是当前计算的调参入口。L4/L5 融合边界分数虽会保存，但高层事件实际由 `aggregate_l4_l5.py` 的连续性合并逻辑构建。

### 检索、评分与问答

下表中的 `config.yaml` 均指 [solution/fully_automatic_track/config.yaml](solution/fully_automatic_track/config.yaml)。

| 参数类别 | 实际生效位置 |
|---|---|
| 环境、模型与接口 | `active_environment_profile` 选择同名的 `environment_profiles`、`model_profiles`，决定数据、输出与本地模型路径；`qa_mllm` 控制 OpenRouter 模型、接口、超时、重试、响应格式和证据数量上限。`day_date_map`、`official_sources.challenge_url` 用于运行前检查。 |
| 查询与逐题策略 | [queries.yaml](solution/fully_automatic_track/queries.yaml) 定义查询、正负提示词、策略配置选择，以及时间、人物、活动和计数规则；先合并题型默认值，再由具体题目字段覆盖。 |
| 召回与区间输出 | `profile_registry.recall_profiles` 控制 top-K、按日期/来源/小时保留候选的数量、文本重排上限和 QA 视觉探测；`profile_registry.output_profiles` 控制 NMS IoU 和基础合并间隔，求解器还会应用逐题合并规则；`submission.sort_interval_rows_by_final_score` 控制 CSV 行序。 |
| EIS/OIS 最终评分 | [solver.py](solution/core_modules/castle_mm26_core/solver.py) 的 `INTERVAL_SCORE_SPECS` 定义逐题视觉、正向文本、负向文本权重，视觉/文本原始分数门槛及必需模态。`_calibrated_threshold` 中最终提交和硬拒绝总分阈值均为 0，主要筛选依据是原始分数门槛和逐题规则。 |
| QA 证据筛选 | `profile_registry.scoring_profiles.qa_evidence.evidence_weights` 定义重排、视觉、负向文本基础权重（0.75/0.25/0.50）；[qa.py](solution/core_modules/castle_mm26_core/qa.py) 中的 `QA_TEXT_FLOORS`、`QA_KEEP_TOP_LIMITS` 及相关函数定义逐题门槛、保留数量和策略附加权重。 |
| 查询阶段模型推理 | [runtimes.py](solution/core_modules/castle_mm26_core/runtimes.py) 定义推理默认值：TIPSv2 为 448/FP32，Qwen3 查询编码为 8192/BF16，BGE-M3 查询编码为 8192/FP16，Qwen3 重排为最大长度 8192、填充长度 1536、BF16。[pipeline.py](solution/core_modules/castle_mm26_core/pipeline.py) 的 `_execution_limits` 控制正式/冒烟运行的限制与批量，正式文本/视觉重排批量为 4/8；`assign_model_devices` 根据 CLI `--devices` 分配设备。 |

尤其需要区分：YAML 中区间评分的 `final_weights`、`object_track_weights` 和 `scoring_component_profiles` 不决定当前 EIS/OIS 最终分数；`submit_quantile`、`min_submit_floor`、`tail_audit_floor` 也不决定最终提交阈值，`hard_floor` 用于判断是否扩展重排候选上限。部分保留字段仍参与配置哈希或按策略名称选择的行为，因此“不参与最终评分”不等于可以随意删除。OpenRouter 请求未设置 `temperature`、`max_tokens`，二者使用服务端默认值。

## 使用前准备

建议使用 Python 3.11、FFmpeg/FFprobe 和兼容 CUDA 的 PyTorch，按各模块的 `requirements.txt` 或 `pyproject.toml` 安装依赖；三套特征编码应使用独立环境。数据、模型权重及运行产物不随代码提供，需先准备完整的本地 TIPSv2、Qwen3-Embedding-8B、BGE-M3 和 Qwen3-Reranker-8B 模型，QA 另需 `OPENROUTER_API_KEY`。

运行前应将下载器、TIPSv2 和 solution 配置中保留的机器绝对路径改为实际位置。下载代理默认指向 `127.0.0.1:7897`；不用代理时，应在下载 YAML 中将 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 显式设为空字符串，直接删除键会恢复代码默认值。

依次完成数据准备、三路编码和事件语料构建并检查验证报告，再通过 [solution/fully_automatic_track/run.py](solution/fully_automatic_track/run.py) 执行运行前检查、EIS、OIS、QA 证据导出、QA 回答和合并打包。各阶段应使用一致的配置和运行标识；修改代码、查询、模型或有效参数后，应使用新的运行标识重新生成受影响的结果。

比赛规则允许参赛者 “receive queries in advance and generate results using any method they choose”，并说明 “participants receive queries in advance and produce their results by whatever process they see fit”。由于比赛对求解方式限制较少，我们针对特定题目进行了优化，降低了系统的泛化能力，因此本系统不太适用于一般问题。
