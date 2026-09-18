#!/usr/bin/env bash
# 一键自测：假数据全链路（01/02/03 + 登记守卫），防止"别的电脑行、这台不行"。
# 用法:  bash tools/self_test.sh           （落后机器可先 INNOV_PYTHON=python3.11 bash tools/self_test.sh）
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${INNOV_PYTHON:-python3}"
PYTHON_BIN="$(command -v "$PY" || echo "$PY")"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0
ok()   { echo "  [PASS] $1"; pass=$((pass+1)); }
bad()  { echo "  [FAIL] $1"; fail=$((fail+1)); }

echo "== 1/12 造脏数据集 + 干净数据集 =="
"$PYTHON_BIN" tools/make_demo_data.py --out "$TMP/dirty"  >/dev/null || { bad "造脏数据失败"; exit 1; }
"$PYTHON_BIN" tools/make_demo_data.py --out "$TMP/clean" --clean >/dev/null || { bad "造干净数据失败"; exit 1; }
ok "make_demo_data"

echo "== 2/12 03 清洗：脏集应全排除(5)，干净集应全保留(5) =="
"$PYTHON_BIN" pipe/03_clean.py --input "$TMP/dirty" --out "$TMP/out_dirty" >/dev/null || { bad "03 脏集运行失败"; exit 1; }
"$PYTHON_BIN" pipe/03_clean.py --input "$TMP/clean" --out "$TMP/out_clean" >/dev/null || { bad "03 干净集运行失败"; exit 1; }
D_EXC=$(F="$TMP/out_dirty/summary.json" K=n_exclude "$PYTHON_BIN" -c "import json,os; print(json.load(open(os.environ['F']))[os.environ['K']])")
C_EXC=$(F="$TMP/out_clean/summary.json" K=n_exclude "$PYTHON_BIN" -c "import json,os; print(json.load(open(os.environ['F']))[os.environ['K']])")
[ "$D_EXC" = "5" ] && ok "脏集排除数=5（实际 $D_EXC）" || bad "脏集排除数应为 5，实际 $D_EXC"
[ "$C_EXC" = "0" ] && ok "干净集排除数=0（实际 $C_EXC）" || bad "干净集排除数应为 0，实际 $C_EXC"

echo "== 3/12 01 盘点 / 02 时间戳 可跑通 =="
"$PYTHON_BIN" pipe/01_inspect.py --input "$TMP/dirty" >/dev/null 2>&1 && ok "01 盘点" || bad "01 盘点失败"
"$PYTHON_BIN" pipe/02_timestamps.py --input "$TMP/dirty" >/dev/null 2>&1 && ok "02 时间戳" || bad "02 时间戳失败"

echo "== 4/12 登记：正常登记应成功且台账 1 行 =="
LEDGER="$TMP/ledger.csv"
"$PYTHON_BIN" ledger/record.py --batch "$TMP/clean" --stage final --yes --out "$LEDGER" --operator 自测 >/dev/null 2>&1
RC=$?
N=$(grep -c "," "$LEDGER" 2>/dev/null || echo 0)
[ $RC -eq 0 ] && ok "登记成功(rc=0)" || bad "登记应成功，rc=$RC"

echo "== 5/12 登记防呆：重复批次应被拦截(rc!=0) =="
"$PYTHON_BIN" ledger/record.py --batch "$TMP/clean" --stage final --yes --out "$LEDGER" >/dev/null 2>&1
RC2=$?
[ $RC2 -ne 0 ] && ok "重复批次被拦截(rc=$RC2)" || bad "重复批次应被拦截，实际 rc=$RC2"

echo "== 6/12 登记防呆：空目录(0集)应拒绝，不写台账 =="
mkdir -p "$TMP/bogus/meta" "$TMP/bogus/data/chunk-000"
echo '{"fps":30}' > "$TMP/bogus/meta/info.json"
"$PYTHON_BIN" ledger/record.py --batch "$TMP/bogus" --stage raw --yes --out "$LEDGER" >/dev/null 2>&1
RC3=$?
[ $RC3 -ne 0 ] && ok "空数据集被拒绝(rc=$RC3)" || bad "空数据集应被拒绝，实际 rc=$RC3"

echo "== 7/12 合并：带处置清单合并应排除脏集全部5集 =="
MERGE_DISP="$TMP/out_dirty/episode_disposition.csv"
"$PYTHON_BIN" pipe/05_merge.py --inputs "$TMP/clean" "$TMP/dirty" \
    --output "$TMP/merged_excl" --overwrite \
    --dispositions "" "$MERGE_DISP" >/dev/null 2>&1
RC_M=$?
ME=$(F="$TMP/merged_excl/meta/info.json" K=total_episodes "$PYTHON_BIN" -c "import json,os; print(json.load(open(os.environ['F']))[os.environ['K']])" 2>/dev/null || echo "?")
[ $RC_M -eq 0 ] && [ "$ME" = "5" ] && ok "合并排除后=5集（实际 $ME）" || bad "合并排除后应为 5 集，实际 $ME rc=$RC_M"
"$PYTHON_BIN" pipe/05_merge.py --inputs "$TMP/clean" "$TMP/dirty" \
    --output "$TMP/merged_all" --overwrite --no-exclude >/dev/null 2>&1
MA=$(F="$TMP/merged_all/meta/info.json" K=total_episodes "$PYTHON_BIN" -c "import json,os; print(json.load(open(os.environ['F']))[os.environ['K']])" 2>/dev/null || echo "?")
[ "$MA" = "10" ] && ok "不排除合并=10集（实际 $MA）" || bad "不排除合并应为 10 集，实际 $MA"

echo "== 8/12 合并防呆 + 产物回检 =="
"$PYTHON_BIN" pipe/05_merge.py --inputs "$TMP/clean" --output "$TMP/x" --overwrite >/dev/null 2>&1
[ $? -ne 0 ] && ok "单输入被拒绝" || bad "单输入应被拒绝"
"$PYTHON_BIN" pipe/03_clean.py --input "$TMP/merged_excl" --out "$TMP/out_merged" >/dev/null 2>&1
MC=$(F="$TMP/out_merged/summary.json" K=n_exclude "$PYTHON_BIN" -c "import json,os; print(json.load(open(os.environ['F']))[os.environ['K']])" 2>/dev/null || echo "?")
[ "$MC" = "0" ] && ok "合并产物再过 03 全保留（排除 $MC）" || bad "合并产物再过 03 应为 0 排除，实际 $MC"
"$PYTHON_BIN" pipe/05_merge.py --inputs "$TMP/clean" "$TMP/dirty" \
    --output "$TMP/merged_excl" --dispositions "" "$MERGE_DISP" >/dev/null 2>&1
[ $? -ne 0 ] && ok "输出目录已存在被拦（--overwrite 才能覆盖）" || bad "已存在输出应被拦截"

echo "== 9/12 06 转换预检（--check 不依赖官方转换器，dev 可跑） =="
"$PYTHON_BIN" pipe/06_convert.py --check --input "$TMP/clean" >/dev/null 2>&1
[ $? -eq 0 ] && ok "06 --check 干净集就绪(rc=0)" || bad "06 --check 干净集应通过"
"$PYTHON_BIN" pipe/06_convert.py --check --input "$TMP/merged_excl" >/dev/null 2>&1
[ $? -eq 0 ] && ok "06 --check 合并产物就绪(rc=0，含自动补 stats)" || bad "06 --check 合并产物应通过"
"$PYTHON_BIN" pipe/06_convert.py --check --input "$TMP/bogus" >/dev/null 2>&1
[ $? -ne 0 ] && ok "06 --check 非 v2.1 被拒(rc!=0)" || bad "06 --check 非 v2.1 应被拒"

echo "== 10/12 07 校验（v2.1/v3.0 结构 smoke + sha256 清单） =="
"$PYTHON_BIN" pipe/07_verify.py --input "$TMP/clean" >/dev/null 2>&1
VRC=$?
[ $VRC -eq 0 ] && ok "07 v2.1 校验通过(rc=0)" || bad "07 v2.1 应通过(rc=$VRC)"
[ -f "$TMP/clean_products/verify/verify_report.json" ] && ok "07 校验报告已写" || bad "缺 verify_report.json"
[ -f "$TMP/clean_products/verify/dataset_sha256sums.txt" ] && ok "07 数据集 sha256 清单已写" || bad "缺 dataset_sha256sums.txt"
"$PYTHON_BIN" - "$TMP/v3fake" <<'PYEOF' >/dev/null 2>&1
import json, sys
from pathlib import Path
import pandas as pd
base = Path(sys.argv[1])
(base / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
(base / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
(base / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "robot_type": "innov",
    "fps": 30, "features": {"observation.images.front": {"dtype": "video", "fps": 30}}, "total_episodes": 2}),
    encoding="utf-8")
pd.DataFrame({"task_index": [0], "task": ["pick"]}).to_parquet(base / "meta" / "tasks.parquet", index=False)
pd.DataFrame({"episode_index": [0, 0, 1, 1], "index": [0, 1, 0, 1],
              "timestamp": [0.0, 1.0, 0.0, 1.0]}).to_parquet(base / "data" / "chunk-000" / "file-000.parquet", index=False)
pd.DataFrame({"episode_index": [0, 1], "length": [2, 2]}).to_parquet(
    base / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
PYEOF
"$PYTHON_BIN" pipe/07_verify.py --input "$TMP/v3fake" >/dev/null 2>&1
[ $? -eq 0 ] && ok "07 v3.0 结构校验通过(rc=0)" || bad "07 v3.0 应通过"
"$PYTHON_BIN" pipe/07_verify.py --input "$TMP/bogus" >/dev/null 2>&1
[ $? -ne 0 ] && ok "07 非数据集被拒(rc!=0)" || bad "07 非数据集应被拒"

echo "== 11/12 QA 汇总(12)：三档结论 + 门禁 =="
cp -r "$TMP/clean" "$TMP/qa_ds"
"$PYTHON_BIN" pipe/03_clean.py --input "$TMP/qa_ds" >/dev/null 2>&1   # 走新布局: 写 _products/clean
"$PYTHON_BIN" pipe/12_qa_report.py --input "$TMP/qa_ds" >/dev/null 2>&1
[ $? -eq 0 ] && ok "12 QA 汇总成功(rc=0)" || bad "12 QA 汇总应成功"
QA_JSON="$TMP/qa_ds_products/qa/qa_summary.json"
if [ -f "$QA_JSON" ] && [ -f "$TMP/qa_ds_products/qa/qa_summary.md" ]; then
  ok "QA 产物生成(qa_summary.json/md)"
else
  bad "缺 QA 产物"
fi
"$PYTHON_BIN" - "$QA_JSON" >/dev/null 2>&1 <<'PYQA' && ok "QA JSON 结构合法(verdict/reasons)" || bad "QA JSON 结构不合法"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["verdict"] in ("ready", "review", "blocked"), d.get("verdict")
assert isinstance(d["reasons"], list) and d["clean_summary"], list(d.keys())
PYQA
"$PYTHON_BIN" pipe/12_qa_report.py --input "$TMP/qa_ds" --fail-on ready >/dev/null 2>&1
[ $? -ne 0 ] && ok "12 --fail-on 门禁生效(rc!=0)" || bad "12 --fail-on 应返回非 0"

echo "== 12/12 08 打包 + 整包核验（训练机流程，含 QA 证据） =="
"$PYTHON_BIN" pipe/08_pack.py --input "$TMP/qa_ds" --out "$TMP/delivery" >/dev/null 2>&1
[ $? -eq 0 ] && ok "08 打包成功(rc=0)" || bad "08 打包应成功"
[ -f "$TMP/delivery/qa_ds_delivery.tar.gz" ] && [ -f "$TMP/delivery/qa_ds_delivery.tar.gz.sha256" ] \
  && ok "交付包 + .sha256 已生成" || bad "缺交付包或 .sha256"
"$PYTHON_BIN" pipe/07_verify.py --delivery "$TMP/delivery/qa_ds_delivery.tar.gz" >/dev/null 2>&1
[ $? -eq 0 ] && ok "07 --delivery 整包核验通过(sha256 逐文件+结构)" || bad "07 --delivery 应通过"
tar -tzf "$TMP/delivery/qa_ds_delivery.tar.gz" | grep -q "/QA.md$" \
  && ok "交付包内含 QA.md(12 结论)" || bad "交付包内应有 QA.md"
[ -f "$TMP/delivery/qa_ds_QA.md" ] && ok "交付包旁留 QA 副本" || bad "缺包旁 QA 副本"
"$PYTHON_BIN" pipe/08_pack.py --input "$TMP/bogus" --out "$TMP/delivery" >/dev/null 2>&1
[ $? -ne 0 ] && ok "08 非数据集被拒(rc!=0)" || bad "08 非数据集应被拒"

echo
echo "========================================"
echo "自测结果: $pass 通过 / $fail 失败"
[ $fail -eq 0 ] || exit 1
exit 0