# innov-dataprep

机器人采集数据的**数据处理 + 数据登记**流水线（面向 LeRobot v2.1/v3.0 数据集，适配 innov / ARX 机械臂）。

- **数据处理模块**：`pipe/`，盘点 → 时间戳审计 →（清洗 → 标注 → 合并 → 转换 → 校验 → 打包，逐步补齐）
- **数据管理模块**：`ledger/`，每批采集/处理完生成"登记卡" → 写台账 `data_catalog.csv` → 多机台账汇总

全部是命令行脚本，**克隆即用**：`git clone` → `bash setup.sh` → `cp config.example.yaml config.yaml` 改路径 → `python3 run.py` 菜单式操作。

> 目前实现：`run.py`（总入口：菜单/短命令、扫批次、状态跟踪、操作留痕）、`01_inspect`（盘点）、`02_timestamps`（时间戳审计）、`03_clean`（清洗/质检，软标记）、`05_merge`（合并，按 03 处置清单排除坏集）、`06_convert`（v2.1→v3.0，调用官方转换器，自动留 v2.1 备份）、`09_annotate`（VLM 逐集评分建议，config 门禁）、`web/`（本地 Web：回放/盲审/台账/一键动作）、`ledger/record.py`（登记卡）、`ledger/aggregate.py`（台账汇总）、`tools/make_demo_data.py`（假数据自测）、`tools/check_config.py`（配置体检）、`tools/self_test.sh`（一键自测）。
> 后续：`07_verify → 08_pack`（规划中）。01/02/03/登记/合并/标注均已支持 v2.1 **和 v3.0**；处理产物统一收进原始目录旁的 `<名字>_products/{阶段}/`。Web 启动：`python3 web/app.py`（默认 127.0.0.1:3100，`--port` 可改）。

---

## 1. 定位（四台机器）

| 机器 | 角色 | 跑什么 |
|---|---|---|
| 采集机 A | robodeploy 采集 + 批次处理 + 合并 + 转 v3.0 | `run.py`（01/02/03 + 05/06/07/08） |
| 训练机 C | 接收交付数据直接训练 | `07_verify --delivery` |
| 第四台电脑 | 台账汇总 | `ledger/aggregate.py` |
| 本机 | 开发 | 仓库开发 / 假数据自测 |

**数据怎么处理的（你的规则）**：想处理哪批就处理哪批、想合并哪些就合并哪些——处理顺序**每批采完立即轻检查**（01/02/03，只报告不删数据）→ **你显式指定**的若干个批次合并成一个集 → 合并后统一标注/清洗 → 转 v3.0 → 校验 → 打包交付训练机。

## 2. 快速开始

```bash
# 采集机上（一次性）
git clone git@github.com:liyang666-dve/innov-dataprep.git   # 公开后也可用 https 链接克隆
cd innov-dataprep
bash setup.sh                        # 自动复用 lerobot conda 环境 / 建 .venv；装依赖
cp config.example.yaml config.yaml   # 只用改 paths 3 个路径，其他保持默认（见"配置速查表"）
bash tools/self_test.sh              # 一键自测（推荐，防止"别台电脑行、这台不行"）

# 日常操作：一条命令总入口（推荐，不用记长命令）
python3 run.py                       # 菜单：列批次 → 选动作 → 选批次编号，全程点选
python3 run.py list                  # 只看批次+状态
python3 run.py clean 1,2             # 清洗批次 1、2（编号见 list）
python3 run.py merge 1,2,3           # 想合并哪些就合并哪些（任意勾选 ≥2 批）
python3 run.py convert 4             # 转 v3.0（选中输出组的合并产物，自动留 v2.1 备份）
python3 run.py record 3              # 登记批次 3（数据处理达标后）
python3 run.py annotate 1,2          # VLM 逐集质量评分+建议（需 config.yaml 配好 annotate 段与 API Key）

# 或直接调底层脚本（精细控制时）
python3 pipe/03_clean.py --input /home/arx/robodeploy/output/arx/arx_0901_1500 --blur
python3 pipe/05_merge.py --inputs 目录A 目录B 目录C --output 合并名   # 显式指定合并
python3 web/app.py                   # 本地 Web：回放/盲审/台账/一键动作（默认 127.0.0.1:3100）
```

**扫描范围**：`run.py` 只扫 `config.yaml → paths.batches` 目录下的**一层子目录**（不会全机器扫）；想扫别处 `python3 run.py list --path <目录>`。

## 3. 配置速查表（只有这里需要你看）

| 配置 | 什么意思 | 要不要改 |
|---|---|---|
| `paths.batches` | 你的批次数据放在哪个目录（run.py 只扫这里） | ✅ **唯一真正要改** |
| `paths.output` | 处理输出、状态文件放哪 | 有默认，可不动 |
| `paths.ledger` | 台账 csv 写哪 | 有默认，可不动 |
| `robot_type_map` | 数据机型编号 → 台账机型简称 | 不填则记 `unk`，不阻塞 |
| `defaults.*` | 登记卡预填值（采集机/操作员/版本/帧率/任务） | 登记时还能改，量力而为 |
| `qc.*` | 03 清洗阈值 | 默认合理，日常不碰 |
| `annotate` / `merge` | 未实现功能的占位 | 忽略 |

> 判断标准：**跑不起来/扫不到数据 → 只可能是 paths.batches 写错了**；其他字段都有默认值。

## 3. 目录结构

```
innov-dataprep/
├── run.py                      # 总入口：交互菜单 / 短命令 / 扫批次 / 状态跟踪 / 操作留痕
├── pipe/                      # 数据处理模块
│   ├── lib/dataset_io.py      # LeRobot v2.1/v3.0 识别/摘要/日期解析 + 产物夹布局（_products/）
│   ├── lib/video_utils.py     # ffprobe 帧数/分辨率（免 PyAV）
│   ├── lib/report.py          # md/csv/json 报告写出
│   ├── lib/suggest.py         # VLM 标注引擎（OpenAI 兼容接口 + config 门禁）
│   ├── lib/replay.py          # 视频/Rerun 回放（Web 用）
│   ├── 01_inspect.py          # 盘点：逐集帧数/时长/帧率/时间戳/NaN/视频对齐
│   ├── 02_timestamps.py       # 时间戳审计：单调性/重复/丢帧窗口（只报告不改数据）
│   ├── 03_clean.py            # 清洗/质检：7+ 项检查，软标记 keep/exclude（只读不删数据）
│   ├── 05_merge.py            # 合并：显式指定 ≥2 批，按 03 处置清单排除坏集，自动命名
│   ├── 06_convert.py          # 转换 v2.1→v3.0：调官方转换器（自动探测调用方式/留备份/补 stats）
│   ├── 09_annotate.py         # 标注：VLM 逐集质量评分+建议（只读；config 门禁拦截）
│   └── migrate_products.py    # 一次性迁移：旧平铺产物（_clean/ 等）收进 _products/（--dry-run 预览）
├── web/                       # 本地 Web 界面（python3 web/app.py，默认 127.0.0.1:3100）
│   ├── app.py                 # Flask 入口
│   └── backend.py             # 与 run.py 共用 execute_action 的动作执行/回放/台账 API
├── ledger/                    # 数据管理模块（登记）
│   ├── record.py              # 登记卡 → 台账 data_catalog.csv（默认 final 阶段：处理达标后）
│   └── aggregate.py           # 多机台账合并汇总
├── tools/
│   ├── make_demo_data.py      # 生成假 v2.1 数据集（带刻意脏数据）供自测
│   ├── check_config.py        # 配置体检（YAML 解析 + 字段结构校验）
│   └── self_test.sh           # 一键自测：假数据全链路 + 登记守卫
├── config.example.yaml        # 配置模板 → 拷成 config.yaml
├── setup.sh                   # 依赖安装（conda env → .venv → 系统 python 兜底）
├── requirements.txt / pyproject.toml
└── LICENSE (Apache-2.0)
```

**处理产物布局**：各阶段产物统一收进数据集旁唯一产品夹 `<名字>_products/{阶段}/`（inspect / timestamps / clean / annotation）；旧的平铺布局（`<名字>_inspect/` 等）仍可读（向后兼容），可跑 `pipe/migrate_products.py` 一次性收拢。`run.py` 扫描时自动跳过 `_products` 与 `_old`（06 转换的 v2.1 备份）。

## 4. 输入数据约定（v2.1）

脚本只识别标准 LeRobot v2.1 布局：

```
<dataset>/
├── meta/info.json            # robot_type / fps / features / videos(分辨率)
├── meta/tasks.jsonl          # {"task_index":0,"task":"..."}
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/<cam>/episode_000000.mp4
```

## 5. 合并（05）与登记（时机与字段）

**合并（想合并哪些就合并哪些）**：`05_merge.py`（或 run.py 菜单"4 合并"）把**你勾选**的 ≥2 个 v2.1 批次合并成一份：
- 自动按各批**清洗处置清单**（`<批>_products/clean/episode_disposition.csv`，兼容旧 `<批>_clean/`）**排除坏集**；没有清单 → 全并入 + `[WARN]`；
- 合并内核与你已验证的 merge_lerobot_v21_arx_bimanual.py 一致（重编号/index/task_index 重写、视频直拷、meta 全套重写）；
- 安全校验：机型/帧率/features/chunks 不一致会拒绝合并；
- 输出默认按公式自动命名（`{task}_{robot}_{首尾日期}_{N}cam_v{ver}`），也可 `--output` 指定；目录已存在会拦截（`--overwrite` 覆盖）；
- 合并结果建议再过一次 `03_clean`（最终清洗）再进 06 转换。

**转换 v2.1→v3.0（06，采集机跑）**：`06_convert.py`（或 run.py 菜单"5 转换"）调用**官方转换器** `convert_dataset_v21_to_v30.py`：
- 自动探测调用方式（`python -m lerobot.scripts.convert_dataset_v21_to_v30` / 包内脚本 / `~/lerobot` 源码目录），**必须在装有 lerobot 的 conda env 运行**（采集机 `lerobot_arx_sdk311`）；
- 本地转换 `--push-to-hub=false`：转换后 v3.0 留在原目录，**原 v2.1 自动备份为 `<名字>_old`**；
- 官方转换器**硬性要求 `meta/episodes_stats.jsonl`**（本仓库 05 合并/假数据/06 预检已保证，缺时 06 自动从 parquet 补算）；
- `--check` 只预检不转换（v2.1? / stats? / 转换器在哪?），采集机上可先 `python3 pipe/06_convert.py --check --input <目录>` 确认就绪再转。

**登记时机（默认）**：数据处理**达标后**登记——即一批数据完成合并、转换、达标后，跑 `record.py`（或 run.py 菜单"6 登记"）对**最终数据集**生成登记卡：
- `--stage final`（默认）：质量记 `clean`，一条台账 = 一个最终数据集；
- `--stage raw`（可选）：对每个原始采集批次登记，质量记 `raw`（保留"每批一行"粒度，不想记就不记，不阻塞主流程）。

**防呆**：非 v2.1/v3.0 或空数据集 → `[ERROR] 拒绝登记`；日期解析不出 → 报错提示 `--date`；批次号重复 → 拦截。

台账字段（对应登记模板）：
`batch_id / task / date / total_days / robot / machine / operator / version / episodes / total_frames / fps / duration_h / avg_duration_min / sensors / format / quality / stage / source / stats / note / registered_at`

命名公式：`{task}_{robot}_{MMDD[-MMDD]}_{N}cam_v{version}`，例：`flosser_innov_0730-0731_3cam_v2`。

几乎全自动推导，人工只需确认 **操作员 / 采集机 / 备注**。

## 6. 开发/自测

```bash
bash tools/self_test.sh              # 一键：假数据 → 01/02/03 → 登记守卫 → PASS/FAIL
python3 tools/check_config.py        # 配置体检
# 本机（依赖在 .pylibs，或建 venv）
PYTHONPATH=/path/to/.pylibs python3 tools/make_demo_data.py --out demo_data/arx_demo_0901_1500
PYTHONPATH=/path/to/.pylibs python3 pipe/03_clean.py --input demo_data/arx_demo_0901_1500
PYTHONPATH=/path/to/.pylibs python3 run.py list --path demo_data
```

## 7. Roadmap

- [x] 总入口 run.py（菜单/短命令/扫批次/状态跟踪/操作留痕）
- [x] 01 盘点 / 02 时间戳审计 / 03 清洗质检（v2.1 + v3.0 均支持，只读软标记）
- [x] 05 合并（显式勾选批次，按 03 处置清单排除坏集，自动命名）
- [x] 06 转换 v2.1→v3.0（官方转换器，本地转换留 v2.1 备份，--check 预检）
- [x] 09 标注（VLM 逐集质量评分+建议，config 门禁拦截；采集机有网可开）
- [x] Web 界面（本地回放/盲审/台账/一键动作，python3 web/app.py）
- [x] 登记（默认处理达标后 final 登记）+ 台账汇总 + 操作留痕
- [x] 产物布局统一（_products/ 唯一产品夹 + 一次性迁移脚本）
- [x] 工具：配置体检 check_config、一键自测 self_test、环境兜底 setup.sh
- [ ] 07 校验（v3.0 加载 smoke + 交付 sha256）
- [ ] 08 打包传输（tar + sha256sums.txt）

## 8. License

Apache 2.0（与 lerobot / robodeploy 一致）。