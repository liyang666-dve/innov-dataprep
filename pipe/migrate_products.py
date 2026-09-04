#!/usr/bin/env python3
"""一次性整理：把旧的平铺产物目录收进新布局 <名字>_products/{阶段}/。

旧布局（散在原始旁的一堆兄弟目录）:
    <名字>_inspect/  <名字>_timestamps/  <名字>_clean/  <名字>_annotation/
新布局（原始旁唯一产品夹）:
    <名字>_products/inspect/  <名字>_products/timestamps/  <名字>_products/clean/ ...
    <名字>_products/annotation/

规则：
  - 只识别 "名字_阶段" 精确后缀，且名字部分是有效数据集（有 meta/info.json）；
  - 目标 _products/{阶段} 若已存在且非空 -> 跳过并警告（不覆盖任何已有内容）；
  - 移到（move）而非复制，成功后原始目录内不再散落；
  - convert 的 <名字>_old 备份属"数据集本体"，不在清理范围（它仍可能有自己的
    _old_clean 等产物，会按同样的规则收进 <名字>_old_products/）。

用法:
    python3 pipe/migrate_products.py --dry-run          # 只列出将移动的
    python3 pipe/migrate_products.py                    # 实际执行（默认扫 config batches+output）
    python3 pipe/migrate_products.py --roots /a /b
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_yaml(path: Path) -> dict:
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return {}


STAGES = ("inspect", "timestamps", "clean", "annotation")


def is_dataset(p: Path) -> bool:
    return (p / "meta" / "info.json").is_file()


def plan(root: Path) -> list[tuple[Path, Path]]:
    """返回 [(来源目录, 目标目录)]，目标 = <base>_products/<stage>（仅当目标为空/不存在）。"""
    moves: list[tuple[Path, Path]] = []
    if not root.is_dir():
        return moves
    for cand in sorted(root.iterdir()):
        if not cand.is_dir():
            continue
        for stage in STAGES:
            suf = f"_{stage}"
            if not cand.name.endswith(suf):
                continue
            base = cand.parent / cand.name[: -len(suf)]
            if not is_dataset(base):
                continue
            dst = base.parent / f"{base.name}_products" / stage
            if dst.exists() and any(dst.rglob("*")):
                print(f"[SKIP] 目标已有内容，不覆盖: {dst}")
                continue
            moves.append((cand, dst))
    return moves


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", default=None,
                    help="要扫描的根目录（默认读 config.yaml 的 paths.batches 与 paths.output）")
    ap.add_argument("--dry-run", action="store_true", help="只列出将移动的目录，不实际移动")
    args = ap.parse_args()

    if args.roots:
        roots = [Path(r).expanduser().resolve() for r in args.roots]
    else:
        cfg = load_yaml(ROOT / "config.yaml")
        paths = cfg.get("paths", {})
        roots = [Path(p).expanduser().resolve() for p in
                 (paths.get("batches"), paths.get("output")) if p]

    all_moves: list[tuple[Path, Path]] = []
    for r in roots:
        all_moves += plan(r)

    if not all_moves:
        print("[i] 没有发现需要迁移的平铺产物目录。")
        return 0

    print(f"共 {len(all_moves)} 项将迁入 _products/：")
    for src, dst in all_moves:
        flag = "[MOVE]" if not args.dry_run else "[计划]"
        print(f"  {flag} {src.parent.name}/{src.name} -> {dst.relative_to(dst.parent.parent)}/")
    if args.dry_run:
        print("[i] --dry-run：以上仅预览，未实际移动。")
        return 0

    for src, dst in all_moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"[OK] {src.name} -> {dst}")
    print("[OK] 迁移完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())