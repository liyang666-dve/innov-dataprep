# innov-dataprep

机器人**采集数据的处理 + 登记**流水线，适用于 LeRobot **v2.1 / v3.0** 数据集（innov / ARX 双臂）。

- 处理：`pipe/` —— 盘点 → 时间戳审计 → 清洗质检 → **第二意见(lerobot-doctor)** → 合并 → VLM 标注 → 转 v3.0 → 校验 → 打包交付
- 登记：`ledger/` —— 每个最终数据集生成"登记卡" → 台账 `data_catalog.csv` → 多机台账汇总

全部是命令行脚本，**克隆即用**：`git clone` → `bash setup.sh` → `cp config.example.yaml config.yaml` 改路径 → `python3 run.py` 菜单式操作。另有本地 Web 界面（回放/盲审/台账/一键动作）。

## 1. 定位（四台机器）

| 机器 | 角色 | 跑什么 |
|---|---|---|
| 采集机 A | robodeploy 采集 + 批次处理 + 合并 + 标注 + 转 v3.0 | `run.py` 全套（01/02/03/05/06/09/登记）+ Web |
| 训练机 C | 接收交付包直接训练 | `sha256sum -c` 或 `07_verify --delivery` 整包核验
| 第四台电脑 | 台账汇总 | `ledger/aggregate.py --dir <各机台账目录>` |
| 本机（开发机） | 仓库开发 | 假数据自测（无需机器人/网络） |

## 2. 快速开始

```bash
git clone https://github.com/liyang666-dve/innov-dataprep.git   # 公开后零认证，任何电脑可用
# （私有阶段或要 push 的机器用 SSH: git clone git@github.com:liyang666-dve/innov-dataprep.git，需配好 GitHub SSH key）
cd innov-dataprep
bash setup.sh                        # 复用 lerobot conda 环境 / 建 .venv / 兜底系统 python3，自动装依赖
cp config.example.yaml config.yaml   # 只需改 paths 3 个路径（见"配置速查表"）
bash tools/self_test.sh              # 一键自测：全绿 = 这台机器环境 OK

# 日常操作（推荐，不用记长命令）
python3 run.py                       # 菜单：列数据 → 选动作 → 选编号，全程点选
python3 run.py list                  # 只看数据+状态（状态 ✓ 表示已跑过哪步）
python3 run.py clean 1,2             # 清洗质检批次 1、2（编号见 list）
python3 run.py doctor 4              # 第二意见：lerobot-doctor 只读体检（需 pip install lerobot-doctor）
python3 run.py merge 1,2,3           # 合并：想合并哪些就勾哪些（≥2 批）
python3 run.py convert 4             # 转 v3.0（输出组的合并产物；自动留 v2.1 备份）
python3 run.py verify 4             # 校验：结构 smoke + 数据集 sha256 清单（转完必跑）
python3 run.py pack 4               # 打包：tar.gz + sha256sums.txt 交付训练机
python3 run.py annotate 1,2          # VLM 逐集质量评分+建议（需配好 annotate 段与 API Key）
python3 run.py record 3              # 登记（数据处理达标后）

# 底层脚本 / Web
python3 pipe/03_clean.py --input <数据集> --blur          # 精细控制时
python3 pipe/07_verify.py --delivery xxx_delivery.tar.gz  # 训练机整包核验（sha256 + 结构）
python3 web/app.py                   # 本地 Web：默认 http://127.0.0.1:8000（端口可用参数改）
```

**扫描范围**：`run.py` 只扫 `paths.batches` 下的**一层子目录**；想扫别处 `python3 run.py list --path <目录>`。`_products`（产物夹）和 `_old`（v2.1 备份）自动跳过。

## 3. 配置速查表（只有这里需要你看）

| 配置 | 什么意思 | 要不要改 |
|---|---|---|
| `paths.batches` | 批次数据目录（run.py 只扫这里） | ✅ **唯一真正要改** |
| `paths.output` / `paths.ledger` | 处理输出 / 台账写哪 | 有默认，可不动 |
| `robot_type_map` | 机型编号 → 台账机型简称 | 不填记 `unk`，不阻塞 |
| `defaults.*` | 登记卡预填（采集机/操作员/版本/帧率/任务） | 登记时还能确认，量力而为 |
| `qc.*` | 03 清洗阈值 | 默认合理，日常不碰 |
| `annotate.*` | VLM 标注：`enabled: true` + `base_url` + `model` + `api_key_env`（环境变量名） | 采集机要标注就配，否则保持 `enabled: false` 自动拦截 |
| `merge.inputs` | 预留占位 | 忽略（合并用命令勾选批次） |

> 判断标准：**跑不起来/扫不到数据 → 基本就是 `paths.batches` 写错了**；其他字段都有默认值。

## 4. 处理与登记流程

**规则**：想处理哪批就处理哪批、想合并哪些就合并哪些。每批采完先**轻检查**（01/02/03，只报告、软标记坏集、**绝不删数据**）→ 你**显式指定**的若干批次**合并**成一个集（自动排除坏集）→ 合并后统一标注/清洗 → **转 v3.0**（v2.1 自动备份 `<名字>_old`）→ **07 校验**（结构 smoke + sha256）→ **12 QA 汇总**（把 01/02/03/07/11 的结论合成一份三档结论）→ **08 打包**（tar.gz 交付，包内带 `QA.md`）→ 训练机核验后直训。

- **输入布局**：标准 v2.1（`meta/` + `data/chunk-*/episode_*.parquet` + `videos/chunk-*/<cam>/*.mp4`）；转换后的 v3.0 同可被 01/02/03/标注/登记处理。
- **合并（05）**：勾选 ≥2 个 v2.1 批次；自动按各批清洗清单（`<批>_products/clean/episode_disposition.csv`，兼容旧 `<批>_clean/`）排除坏集；机型/帧率/features 不一致会拒绝；输出自动命名 `{task}_{robot}_{MMDD[-MMDD]}_{N}cam_v{ver}`，已存在会拦截（`--overwrite` 覆盖）。
- **质检三档判定（03）**：`exclude`（硬伤：文件对不上/NaN/维度不一致 → 05 合并剔除）/ `review`（需人看：常量维、动作尖峰、静止帧占比、时长离群、stats 漂移 → 05 **不剔除**，进 Web 盲审页）/ `keep`。另有 info 级统计信号（僵死维、有效运动比、最大跳变 Top3）只进报告，不参与判定——阈值未定标前绝不杀数据。
- **第二意见（11）**：包装官方生态的 `lerobot-doctor`（只读 `check`）做独立体检——补上自研 03 覆盖不到的 action 级异常（尖峰/僵死/零方差维度/策略兼容性/URDF 动力学/per-episode 明细）。默认把被点名的 `keep` 集降级为 `review`（05 只排除 `exclude`，`review` 不会被删），**绝不自动排除**；也绝不调用 doctor 的 `fix`/`trim`（那两个会改数据）。装不上就让这一步跳过，不影响其它步骤。
- **QA 汇总（12）**：把一次数据集的所有证据（03 逐集判定 + 11 第二意见 + 07 校验 + 02 时间戳 + 交付包）合成 `<ds>_products/qa/qa_summary.{md,json}`，给出**一个结论**：`ready` / `review` / `blocked`（blocked = doctor FAIL / 07 不过 / 没跑 03；review = 有 review 集或 doctor WARN）。`--fail-on review` 可当门禁（非 0 退出）。08 打包会把 `qa_summary.md` 作为 `<ds>/QA.md` 放进交付包并计入 sha256 清单，同时在包旁留一份 `<ds>_QA.md`；登记时 QA 结论写进台账 `stats` 列（如 `93集/70958帧/30fps/0.66h/QA:review(rev2,exc3)`），台账列结构不变。
- **阈值怎么来的 / 怎么自查（2026-09 实测）**：三类来源——① 物理基线（帧率偏差/丢帧比/时长安全网）② 对齐官方生态口径（尖峰 8σ、时长 3σ 离群、stats 漂移 5%）③ **必须用自己数据定标**（关节限位/跳变/卡死、尖峰绝对下限、常量维）。用注入缺陷实测过敏感性：NaN→exclude、常量维→review、9% 丢帧→exclude、视频少帧→exclude、重模糊→exclude、95% 静止→review、stats 漂移→review 都抓得住；同时修掉两个真问题——**关节超限位时曾 IndexError 崩溃**（展平索引当列号用）、**stats 漂移在"逐列键"与 v3.0 存法下静默失效**（现支持三种存法）。单帧跳变加了 `joint_jump_review_rad`（默认 0.8）：定标把排除线放宽到 3.05 后，0.8~3.05 rad 的单帧跳变仍会进 review 而不是被放过。
- **阈值定标（`tools/calibrate_qc.py`，只读）**：03 的关节/尖峰阈值不该拍脑袋——实测你 0730 两批各 15 集得出的建议是 `joint_jump_rad=2.37` / `stuck_s=28.5` / `zero_var_eps=0.006` / `action_spike_min_abs=0.47`，而旧默认 0.8 / 0.4 在真实数据上必然大量误杀。跑 `python3 tools/calibrate_qc.py --dir <批次目录> --max-episodes 0` 出报告 + `qc_suggested.yaml`，人工确认后再粘进 config。
- **转换（06，仅采集机）**：包装官方 `convert_dataset_v21_to_v30.py`（自动探测调用方式）；`--push-to-hub=false` 本地转；官方转换器需要 `meta/episodes_stats.jsonl`，缺时自动补算；`--check` 先预检再转。
- **标注（09）**：VLM（OpenAI 兼容接口，可接 DeepSeek/通义）逐集评分+建议，**只读**；未启用/缺 Key 会明确拦截。
- **登记（默认时机：数据处理达标后）**：`--stage final`（默认，质量 `clean`，一条台账 = 一个最终数据）；`--stage raw`（可选，每原始批次一行）。防呆：非 v2.1/v3.0、空数据集、假日期、批次号重复都会拦截。
- 台账字段：`batch_id / task / date / robot / machine / operator / version / episodes / total_frames / fps / duration_h / avg_duration_min / sensors / format / quality / stage / source / stats / note / registered_at`（几乎全自动推导，人工只需确认操作员/采集机/备注）。

**处理产物布局**：各阶段产物统一收进数据集旁唯一产品夹 `<名字>_products/{阶段}/`（inspect / timestamps / clean / doctor / qa / verify / annotation）；旧平铺布局（`<名字>_inspect/` 等）仍可读，可跑 `pipe/migrate_products.py` 一次性收拢（`--dry-run` 预览）。

## 5. 在别的电脑克隆即用（依赖分档）

`setup.sh` 自动处理一切；下面是"哪些功能需要什么"的对照，方便判断某台机器能跑什么：

| 功能 | 需要 | 说明 |
|---|---|---|
| 01/02/03/05/07/08/合并/登记/台账 | Python 3.10+ 核心包（numpy/pandas/pyarrow/PyYAML） | setup.sh 自动装，**开箱即用**（07/08 只用标准库 tarfile/hashlib + pyarrow 页脚） |
| 视频帧数核对（01/03） | 系统 `ffprobe` | `sudo apt install ffmpeg`（Ubuntu）；setup.sh 只检查提示、不替你装；缺则该项自动跳过 + `[WARN]` |
| 03 `--blur` 模糊检查 | opencv | setup.sh 可选行自动尝试装 |
| 06 转换 v2.1→v3.0 | **lerobot 环境** | **只用采集机能跑**（conda `lerobot_arx_sdk311`）；其他机器会明确报错提示 |
| 09 VLM 标注 | 仅标准库 + 网络 + API Key | 配好 config annotate 段即可 |
| 12 QA 汇总 | 无（纯标准库） | 读上面的产物合成结论；缺哪个阶段就少一行依据，不报错 |
| 11 第二意见 | `lerobot-doctor`（本仓库不打包） | `uv tool install lerobot-doctor` 或 `pip install --user lerobot-doctor`；找不到时报错跳过 |
| Web（回放/盲审/台账） | flask（回放另需 rerun-sdk） | setup.sh 可选行自动装；不跑 Web 可忽略 |

另外三点：**私有仓库**需要各机配 GitHub 认证（SSH key 或 PAT），想零认证可直接把仓库改公开；**Python ≥ 3.10**（Ubuntu 22.04+ 自带）；每台机器第一次 clone 后跑 `self_test.sh`，全绿即环境 OK。

## 6. 目录结构

```
innov-dataprep/
├── run.py                      # 总入口：菜单/短命令/扫批次/状态跟踪/操作留痕
├── pipe/                       # 数据处理
│   ├── lib/                    # dataset_io(识别/摘要/产物布局) video_utils(ffprobe)
│   │                           # report suggest(VLM引擎) replay(回放)
│   ├── 01_inspect.py 02_timestamps.py 03_clean.py   # 轻检查（只读）
│   ├── 05_merge.py 06_convert.py 07_verify.py 08_pack.py 09_annotate.py
│   ├── 11_doctor.py 12_qa_report.py   # 第二意见 / QA 汇总（只读）
│   └── migrate_products.py     # 旧平铺产物 → _products/ 一次性迁移
├── web/                        # Flask 本地界面（app.py 入口 + backend.py API）
├── ledger/record.py aggregate.py   # 登记卡 / 台账汇总
├── tools/make_demo_data.py calibrate_qc.py check_config.py self_test.sh
├── config.example.yaml setup.sh requirements.txt pyproject.toml
└── LICENSE (Apache-2.0)
```

## 7. 开发/自测

```bash
bash tools/self_test.sh              # 假数据全链路 + 登记守卫，一键验证
python3 tools/check_config.py        # 配置体检
# 开发机（依赖在 .pylibs）：PYTHONPATH=/path/to/.pylibs python3 <上面任意命令>
```

## 8. Roadmap

- [x] 01/02/03 轻检查（v2.1+v3.0，只读软标记）· 05 合并 · 06 转换（官方转换器）
- [x] **数组列修复（2026-09）**：真实 v2.1 的 `action`/`observation.state` 是"每行一个数组"的 object 列，旧实现按 dtype/点分前缀选列 → NaN/Inf 与关节类检查在真实数据上不触发。现统一 `expand_array_features()` 展开后再查；关节类判定因无关节名（无法豁免夹爪）默认关闭，只输出「关节观测量」供定标，`--joints` 才参与判定
- [x] 11 第二意见（lerobot-doctor 只读体检，可联动把 keep 降级为 review）
- [x] 09 标注（VLM，config 门禁）· Web 界面（回放/盲审/台账/一键动作）
- [x] 07 校验（结构 smoke + 数据集/交付包 sha256，`--delivery` 整包核验）· 08 打包（tar.gz + sha256sums.txt）
- [x] 登记 + 台账汇总 + 操作留痕 · 产物布局统一（_products/ + 迁移脚本）
- [x] **统计信号 + 三档判定（2026-09）**：常量维/动作尖峰/僵死维/有效运动比/时长离群/stats 漂移内建进 03（不依赖第三方），review 进 Web 盲审页
- [x] 12 QA 汇总（三档结论 + `--fail-on` 门禁）· 交付包内带 `QA.md` · 台账 stats 记 QA 结论 · `tools/calibrate_qc.py` 阈值定标
- [x] 阈值敏感性实测 + 两个真 bug 修复（关节超限位崩溃、stats 漂移在逐列键/v3.0 存法下失效）+ 跳变复核线兜底
- [x] **全流程已实现**（01→08 + QA 闭环；只剩真实数据上的端到端验证与可选 RDA 接入）

## 9. License

Apache 2.0（与 lerobot / robodeploy 一致）。