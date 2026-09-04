#!/usr/bin/env python3
"""08 打包：给训练机出**交付包**（data + videos + 全套 meta）。

打包内容：数据集本体（v2.1/v3.0）整个目录，外加 sha256sums.txt（数据集内每个文件的
sha256 清单，相对路径，训练机可用 07_verify --delivery 整包核验）。

产物（默认放 <数据集> 同级，或用 config paths.output）：
    <名字>_delivery.tar.gz             # 交付包（tar.gz，顶层目录名 = 数据集名）
    <名字>_delivery.tar.gz.sha256      # 交付包本体的 sha256（传输校验用）

用法:
    python3 pipe/08_pack.py --input <数据集> [--out 目录] [--overwrite] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io  # noqa: E402


def sha256_of_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_manifest(ds: Path) -> list[tuple[str, str]]:
    rows = []
    for p in sorted(ds.rglob("*")):
        if p.is_file():
            rows.append((p.relative_to(ds).as_posix(), sha256_of_file(p)))
    return rows


def pack_dataset(ds: Path, out_dir: Path, force: bool = False) -> tuple[Path, Path, int]:
    """打交付包。返回 (tar 路径, tar.sha256 路径, 文件数)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_p = out_dir / f"{ds.name}_delivery.tar.gz"
    if tar_p.exists() and not force:
        raise FileExistsError(f"{tar_p} 已存在（用 --overwrite 覆盖或换 --out）")

    rows = build_manifest(ds)
    with tarfile.open(tar_p, "w:gz") as tf:
        for rel, _ in rows:
            tf.add(ds / rel, arcname=f"{ds.name}/{rel}")
        # 清单放数据集根内（路径不含清单自身，保持纯数据清单）
        manifest_bytes = ("\n".join(f"{h}  {rel}" for rel, h in rows) + "\n").encode("utf-8")
        ti = tarfile.TarInfo(f"{ds.name}/sha256sums.txt")
        ti.size = len(manifest_bytes)
        tf.addfile(ti, __import__("io").BytesIO(manifest_bytes))

    sha_p = out_dir / f"{tar_p.name}.sha256"
    sha_p.write_text(f"{sha256_of_file(tar_p)}  {tar_p.name}\n", encoding="utf-8")
    return tar_p, sha_p, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="08 打包：tar.gz + sha256sums.txt 交付训练机")
    ap.add_argument("--input", required=True, help="数据集目录（v2.1/v3.0）")
    ap.add_argument("--out", help="交付包输出目录（默认数据集同级；run.py 会用 paths.output）")
    ap.add_argument("--overwrite", action="store_true", help="已存在交付包时覆盖")
    ap.add_argument("--force", action="store_true", help="跳过未校验提示（不建议）")
    args = ap.parse_args()

    ds = Path(args.input).expanduser()
    kind, reason = dataset_io.detect_dataset(ds)
    if kind not in ("v2.1", "v3.0"):
        print(f"[ERROR] 不是数据集，拒绝打包: {ds}（{reason}）")
        return 1
    if not args.force:
        vreport = ds.parent / f"{ds.name}_products" / "verify" / "verify_report.json"
        if vreport.is_file():
            import json
            r = json.loads(vreport.read_text(encoding="utf-8"))
            if r.get("ok"):
                print(f"[i] 已通过 07 校验（{r.get('n_episodes')} 集，{r.get('total_frames')} 帧）")
            else:
                print(f"[WARN] 07 校验曾失败（{r.get('n_fail')} 项），建议先跑 07；--force 可跳过")
        else:
            print("[WARN] 未找到 07 校验报告，建议先 python3 pipe/07_verify.py --input <数据集>")
    try:
        tar_p, sha_p, n_files = pack_dataset(ds, Path(args.out).expanduser() if args.out else ds.parent,
                                             force=args.overwrite)
    except FileExistsError as e:
        print(f"[ERROR] {e}")
        return 1

    size_mb = tar_p.stat().st_size / 1_048_576
    print(f"[OK] 交付包已生成: {tar_p}")
    print(f"[OK] 大小 {size_mb:.1f} MB / 数据文件 {n_files} 个 / 校验 {sha_p.name}")
    print(f"\n训练机整包核验（拷到训练机后执行）:")
    print(f"    sha256sum -c {sha_p.name}")
    print(f"    python3 pipe/07_verify.py --delivery {tar_p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())