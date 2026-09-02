# innov-dataprep

机器人采集数据的**数据处理 + 数据登记**流水线（面向 LeRobot v2.1/v3.0 数据集，适配 innov / ARX 机械臂）。

- **数据处理模块**：`pipe/`，盘点 → 时间戳审计 →（清洗 → 标注 → 合并 → 转换 → 校验 → 打包，逐步补齐）
- **数据管理模块**：`ledger/`，每批采集/处理完生成"登记卡" → 写台账 `data_catalog.csv` → 多机台账汇总

全部是命令行脚本，**克隆即用**：`git clone` → `bash setup.sh` → `cp config.example.yaml config.yaml` 改路径 → 按需跑脚本。

> 目前实现：`01_inspect`（盘点）、`02_timestamps`（时间戳审计）、`03_clean`（清洗/质检，软标记）、`ledger/record.py`（登记卡）、`ledger/aggregate.py`（台账汇总）、`tools/make_demo_data.py`（假数据自测）。
> 后续：`04_annotate → 05_merge → 06_convert → 07_verify → 08_pack`（规划中）。

---

## 1. 定位（四台机器）

| 机器 | 角色 | 跑什么 |
|---|---|---|
| 采集机 A | robodeploy 采集 + 批次处理 + 合并 + 转 v3.0 | `01/02/03...` + `05/06/07/08` |
| 训练机 C | 接收交付数据直接训练 | `07_verify --delivery` |
| 第四台电脑 | 台账汇总 | `ledger/aggregate.py` |
| 本机 | 开发 | 仓库开发 / 假数据自测 |

**数据处理顺序（已定）**：每批采完**立即轻检查**（01/02，只报告不删数据）→ 你指定的若干个批次**合并**成一个集 → 合并后统一 **标注/清洗 → 转 v3.0 → 校验 → 打包交付** 训练机。

## 2. 快速开始

```bash
# 采集机上（一次性）
git clone https://github.com/liyang666-dve/innov-dataprep.git
cd innov-dataprep
bash setup.sh                      # 复用 lerobot_arx_sdk311 环境；自研部分只依赖 numpy/pandas/pyarrow
cp config.example.yaml config.yaml # 填你的路径 / 机器人映射 / 默认操作员等

# 每批采集完，立即盘点 + 时间戳审计 + 质检（都是只读、当天发现当天处理）
python3 pipe/01_inspect.py     --input /home/arx/robodeploy/output/arx/arx_0901_1500
python3 pipe/02_timestamps.py  --input /home/arx/robodeploy/output/arx/arx_0901_1500
python3 pipe/03_clean.py       --input /home/arx/robodeploy/output/arx/arx_0901_1500   # 默认不查模糊帧
python3 pipe/03_clean.py       --input /home/arx/robodeploy/output/arx/arx_0901_1500 --blur  # 加查模糊帧(较慢)

# 处理/合并完成后，登记这一批
python3 ledger/record.py --batch /home/arx/robodeploy/output/arx/arx_0901_1500 --operator 张三 --yes

# 攒够一批后，第四台电脑汇总台账
python3 ledger/aggregate.py --dir /path/to/ledgers
```

## 3. 目录结构

```
innov-dataprep/
├── pipe/                      # 数据处理模块
│   ├── lib/dataset_io.py      # LeRobot v2.1 读写/摘要（盘点、登记共用）
│   ├── lib/video_utils.py     # ffprobe 帧数/分辨率（免 PyAV）
│   ├── lib/report.py          # md/csv/json 报告写出
│   ├── 01_inspect.py          # 盘点：逐集帧数/时长/帧率/时间戳/NaN/视频对齐
│   └── 02_timestamps.py       # 时间戳审计：单调性/重复/丢帧窗口（只报告不改数据）
├── ledger/                    # 数据管理模块（登记）
│   ├── record.py              # 登记卡 → 台账 data_catalog.csv（每批一行）
│   └── aggregate.py           # 多机台账合并汇总
├── tools/make_demo_data.py    # 生成假 v2.1 数据集（带刻意脏数据）供自测
├── config.example.yaml        # 配置模板 → 拷成 config.yaml
├── setup.sh                   # 依赖安装（conda env 探测 + pip）
├── requirements.txt / pyproject.toml
└── LICENSE (Apache-2.0)
```

## 4. 输入数据约定（v2.1）

脚本只识别标准 LeRobot v2.1 布局：

```
<dataset>/
├── meta/info.json            # robot_type / fps / features / videos(分辨率)
├── meta/tasks.jsonl          # {"task_index":0,"task":"..."}
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/<cam>/episode_000000.mp4
```

## 5. 台账字段（对应登记模板）

`batch_id / task / date / total_days / robot / machine / operator / version / episodes / total_frames / fps / duration_h / avg_duration_min / sensors / format / quality / source / note / stats / registered_at`

命名公式：`{task}_{robot}_{MMDD[-MMDD]}_{N}cam_v{version}`，例：`flosser_innov_0730-0731_3cam_v2`。

几乎全自动推导，人工只需确认 **操作员 / 采集机 / 备注**。

## 6. 开发/自测

```bash
# 本机（依赖在 .pylibs，或建 venv）
PYTHONPATH=/path/to/.pylibs python3 tools/make_demo_data.py --out demo_data/arx_demo_0901_1500
PYTHONPATH=/path/to/.pylibs python3 pipe/01_inspect.py    --input demo_data/arx_demo_0901_1500
PYTHONPATH=/path/to/.pylibs python3 pipe/02_timestamps.py --input demo_data/arx_demo_0901_1500
PYTHONPATH=/path/to/.pylibs python3 ledger/record.py --batch demo_data/arx_demo_0901_1500 --operator 测试 --yes
```

## 7. Roadmap

- [x] 01 盘点（v2.1）
- [x] 02 时间戳审计（只读）
- [x] 03 清洗/质检（7+ 项检查 + 软标记，只读不删数据）
- [x] 登记卡 + 台账 + 汇总
- [ ] 04 标注（指令模板 + LLM 建议 + 写回 tasks/parquet）
- [ ] 05 合并（官方 merge，按 03 的 episode_disposition.csv 排除坏集，指定文件清单）
- [ ] 06 转换 v2.1→v3.0（采集机，官方 converter）
- [ ] 07 校验（v3.0 加载 smoke + 交付 sha256）
- [ ] 08 打包传输（tar + sha256sums.txt）

## 8. License

Apache 2.0（与 lerobot / robodeploy 一致）。