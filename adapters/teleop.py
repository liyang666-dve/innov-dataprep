"""遥操作透传适配器：robodeploy/ARX 采集机输出已是标准 v2.1 → 直通，不重复转换。

遥操作源是现有链路（01→08）的原始输入，INGEST 对它的职责是：
  1) 校验已是标准 v2.1；
  2) 补写 source_meta 标记（source_type=teleop，动作通道=关节），便于台账溯源；
  3) 不复制大文件（视频），直接原位登记即可。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io  # noqa: E402


def detect(src: Path) -> tuple[bool, str]:
    kind, reason = dataset_io.detect_dataset(Path(src))
    if kind != "v2.1":
        return False, f"非标准 v2.1（{kind}: {reason}）——遥操作源应直接是 LeRobot v2.1"
    return True, "已是标准 v2.1"


def ingest(src: Path, out_root: Path, cfg: dict) -> tuple[Path, dict]:
    """直通：补 source_meta（若缺），返回原路径。"""
    src = Path(src)
    info = dataset_io.read_json(src / "meta" / "info.json")
    if "source_meta" not in info:
        info["source_meta"] = {
            "source_type": "teleop", "adapter": "teleop.v1",
            "action_space": {"kind": "joint", "note": "robodeploy 关节流，走既有 01→08"},
        }
        (src / "meta" / "info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "source": "teleop", "episodes": "n/a（原样）", "action_channel": "joint",
        "output": str(src), "warnings": ["直通：不复制数据，直接走处理链 01→08"],
    }
    return src, report
