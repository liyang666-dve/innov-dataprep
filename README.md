# innov-dataprep

机器人**采集数据的处理 + 登记**流水线，适用于 LeRobot **v2.1 / v3.0** 数据集（innov / ARX 双臂）。

- 处理：`pipe/` —— 盘点 → 时间戳审计 → 清洗质检 → 合并 → VLM 标注 → 转 v3.0 → 校验 → 打包交付
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
git clone git@github.com:liyang666-dve/innov-dataprep.git   # 私有仓库需本机配好 GitHub 认证；公开后零认证
cd innov-dataprep
bash setup.sh                        # 复用 lerobot conda 环境 / 建 .venv / 兜底系统 python3，自动装依赖
cp config.example.yaml config.yaml   # 只需改 paths 3 个路径（见"配置速查表"）
bash tools/self_test.sh              # 一键自测：全绿 = 这台机器环境 OK

# 日常操作（推荐，不用记长命令）
python3 run.py                       # 菜单：列数据 → 选动作 → 选编号，全程点选
python3 run.py list                  # 只看数据+状态（状态 ✓ 表示已跑过哪步）
python3 run.py clean 1,2             # 清洗质检批次 1、2（编号见 list）
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

**规则**：想处理哪批就处理哪批、想合并哪些就合并哪些。每批采完先**轻检查**（01/02/03，只报告、软标记坏集、**绝不删数据**）→ 你**显式指定**的若干批次**合并**成一个集（自动排除坏集）→ 合并后统一标注/清洗 → **转 v3.0**（v2.1 自动备份 `<名字>_old`）→ **07 校验**（结构 smoke + sha256）→ **08 打包**（tar.gz 交付）→ 训练机核验后直训。

- **输入布局**：标准 v2.1（`meta/` + `data/chunk-*/episode_*.parquet` + `videos/chunk-*/<cam>/*.mp4`）；转换后的 v3.0 同可被 01/02/03/标注/登记处理。
- **合并（05）**：勾选 ≥2 个 v2.1 批次；自动按各批清洗清单（`<批>_products/clean/episode_disposition.csv`，兼容旧 `<批>_clean/`）排除坏集；机型/帧率/features 不一致会拒绝；输出自动命名 `{task}_{robot}_{MMDD[-MMDD]}_{N}cam_v{ver}`，已存在会拦截（`--overwrite` 覆盖）。
- **转换（06，仅采集机）**：包装官方 `convert_dataset_v21_to_v30.py`（自动探测调用方式）；`--push-to-hub=false` 本地转；官方转换器需要 `meta/episodes_stats.jsonl`，缺时自动补算；`--check` 先预检再转。
- **标注（09）**：VLM（OpenAI 兼容接口，可接 DeepSeek/通义）逐集评分+建议，**只读**；未启用/缺 Key 会明确拦截。
- **登记（默认时机：数据处理达标后）**：`--stage final`（默认，质量 `clean`，一条台账 = 一个最终数据）；`--stage raw`（可选，每原始批次一行）。防呆：非 v2.1/v3.0、空数据集、假日期、批次号重复都会拦截。
- 台账字段：`batch_id / task / date / robot / machine / operator / version / episodes / total_frames / fps / duration_h / avg_duration_min / sensors / format / quality / stage / source / stats / note / registered_at`（几乎全自动推导，人工只需确认操作员/采集机/备注）。

**处理产物布局**：各阶段产物统一收进数据集旁唯一产品夹 `<名字>_products/{阶段}/`（inspect / timestamps / clean / annotation）；旧平铺布局（`<名字>_inspect/` 等）仍可读，可跑 `pipe/migrate_products.py` 一次性收拢（`--dry-run` 预览）。

## 5. 在别的电脑克隆即用（依赖分档）

`setup.sh` 自动处理一切；下面是"哪些功能需要什么"的对照，方便判断某台机器能跑什么：

| 功能 | 需要 | 说明 |
|---|---|---|
| 01/02/03/05/07/08/合并/登记/台账 | Python 3.10+ 核心包（numpy/pandas/pyarrow/PyYAML） | setup.sh 自动装，**开箱即用**（07/08 只用标准库 tarfile/hashlib + pyarrow 页脚） |
| 视频帧数核对（01/03） | 系统 `ffprobe` | `sudo apt install ffmpeg`（Ubuntu）；setup.sh 只检查提示、不替你装；缺则该项自动跳过 + `[WARN]` |
| 03 `--blur` 模糊检查 | opencv | setup.sh 可选行自动尝试装 |
| 06 转换 v2.1→v3.0 | **lerobot 环境** | **只用采集机能跑**（conda `lerobot_arx_sdk311`）；其他机器会明确报错提示 |
| 09 VLM 标注 | 仅标准库 + 网络 + API Key | 配好 config annotate 段即可 |
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
│   └── migrate_products.py     # 旧平铺产物 → _products/ 一次性迁移
├── web/                        # Flask 本地界面（app.py 入口 + backend.py API）
├── ledger/record.py aggregate.py   # 登记卡 / 台账汇总
├── tools/make_demo_data.py check_config.py self_test.sh
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
- [x] 09 标注（VLM，config 门禁）· Web 界面（回放/盲审/台账/一键动作）
- [x] 07 校验（结构 smoke + 数据集/交付包 sha256，`--delivery` 整包核验）· 08 打包（tar.gz + sha256sums.txt）
- [x] 登记 + 台账汇总 + 操作留痕 · 产物布局统一（_products/ + 迁移脚本）
- [x] **全流程已实现**（01→08 闭环；只剩真实数据上的阈值定标与端到端验证）

## 9. License

Apache 2.0（与 lerobot / robodeploy 一致）。