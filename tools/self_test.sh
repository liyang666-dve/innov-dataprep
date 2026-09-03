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

echo "== 1/6 造脏数据集 + 干净数据集 =="
"$PYTHON_BIN" tools/make_demo_data.py --out "$TMP/dirty"  >/dev/null || { bad "造脏数据失败"; exit 1; }
"$PYTHON_BIN" tools/make_demo_data.py --out "$TMP/clean" --clean >/dev/null || { bad "造干净数据失败"; exit 1; }
ok "make_demo_data"

echo "== 2/6 03 清洗：脏集应全排除(5)，干净集应全保留(5) =="
"$PYTHON_BIN" pipe/03_clean.py --input "$TMP/dirty" --out "$TMP/out_dirty" >/dev/null || { bad "03 脏集运行失败"; exit 1; }
"$PYTHON_BIN" pipe/03_clean.py --input "$TMP/clean" --out "$TMP/out_clean" >/dev/null || { bad "03 干净集运行失败"; exit 1; }
D_EXC=$("$PYTHON_BIN" -c "import json,sys; print(json.load(open('$TMP/out_dirty/summary.json'))['n_exclude'])")
C_EXC=$("$PYTHON_BIN" -c "import json,sys; print(json.load(open('$TMP/out_clean/summary.json'))['n_exclude'])")
[ "$D_EXC" = "5" ] && ok "脏集排除数=5（实际 $D_EXC）" || bad "脏集排除数应为 5，实际 $D_EXC"
[ "$C_EXC" = "0" ] && ok "干净集排除数=0（实际 $C_EXC）" || bad "干净集排除数应为 0，实际 $C_EXC"

echo "== 3/6 01 盘点 / 02 时间戳 可跑通 =="
"$PYTHON_BIN" pipe/01_inspect.py --input "$TMP/dirty" >/dev/null 2>&1 && ok "01 盘点" || bad "01 盘点失败"
"$PYTHON_BIN" pipe/02_timestamps.py --input "$TMP/dirty" >/dev/null 2>&1 && ok "02 时间戳" || bad "02 时间戳失败"

echo "== 4/6 登记：正常登记应成功且台账 1 行 =="
LEDGER="$TMP/ledger.csv"
"$PYTHON_BIN" ledger/record.py --batch "$TMP/clean" --stage final --yes --out "$LEDGER" --operator 自测 >/dev/null 2>&1
RC=$?
N=$(grep -c "," "$LEDGER" 2>/dev/null || echo 0)
[ $RC -eq 0 ] && ok "登记成功(rc=0)" || bad "登记应成功，rc=$RC"

echo "== 5/6 登记防呆：重复批次应被拦截(rc!=0) =="
"$PYTHON_BIN" ledger/record.py --batch "$TMP/clean" --stage final --yes --out "$LEDGER" >/dev/null 2>&1
RC2=$?
[ $RC2 -ne 0 ] && ok "重复批次被拦截(rc=$RC2)" || bad "重复批次应被拦截，实际 rc=$RC2"

echo "== 6/6 登记防呆：空目录(0集)应拒绝，不写台账 =="
mkdir -p "$TMP/bogus/meta" "$TMP/bogus/data/chunk-000"
echo '{"fps":30}' > "$TMP/bogus/meta/info.json"
"$PYTHON_BIN" ledger/record.py --batch "$TMP/bogus" --stage raw --yes --out "$LEDGER" >/dev/null 2>&1
RC3=$?
[ $RC3 -ne 0 ] && ok "空数据集被拒绝(rc=$RC3)" || bad "空数据集应被拒绝，实际 rc=$RC3"

echo "== 7/8 合并：带处置清单合并应排除脏集全部5集 =="
MERGE_DISP="$TMP/out_dirty/episode_disposition.csv"
"$PYTHON_BIN" pipe/05_merge.py --inputs "$TMP/clean" "$TMP/dirty" \
    --output "$TMP/merged_excl" --overwrite \
    --dispositions "" "$MERGE_DISP" >/dev/null 2>&1
RC_M=$?
ME=$("$PYTHON_BIN" -c "import json,sys; print(json.load(open('$TMP/merged_excl/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
[ $RC_M -eq 0 ] && [ "$ME" = "5" ] && ok "合并排除后=5集（实际 $ME）" || bad "合并排除后应为 5 集，实际 $ME rc=$RC_M"
"$PYTHON_BIN" pipe/05_merge.py --inputs "$TMP/clean" "$TMP/dirty" \
    --output "$TMP/merged_all" --overwrite --no-exclude >/dev/null 2>&1
MA=$("$PYTHON_BIN" -c "import json,sys; print(json.load(open('$TMP/merged_all/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
[ "$MA" = "10" ] && ok "不排除合并=10集（实际 $MA）" || bad "不排除合并应为 10 集，实际 $MA"

echo "== 8/8 合并防呆 + 产物回检 =="
"$PYTHON_BIN" pipe/05_merge.py --inputs "$TMP/clean" --output "$TMP/x" --overwrite >/dev/null 2>&1
[ $? -ne 0 ] && ok "单输入被拒绝" || bad "单输入应被拒绝"
"$PYTHON_BIN" pipe/03_clean.py --input "$TMP/merged_excl" --out "$TMP/out_merged" >/dev/null 2>&1
MC=$("$PYTHON_BIN" -c "import json,sys; print(json.load(open('$TMP/out_merged/summary.json'))['n_exclude'])" 2>/dev/null || echo "?")
[ "$MC" = "0" ] && ok "合并产物再过 03 全保留（排除 $MC）" || bad "合并产物再过 03 应为 0 排除，实际 $MC"
"$PYTHON_BIN" pipe/05_merge.py --inputs "$TMP/clean" "$TMP/dirty" \
    --output "$TMP/merged_excl" --dispositions "" "$MERGE_DISP" >/dev/null 2>&1
[ $? -ne 0 ] && ok "输出目录已存在被拦（--overwrite 才能覆盖）" || bad "已存在输出应被拦截"

echo
echo "========================================"
echo "自测结果: $pass 通过 / $fail 失败"
[ $fail -eq 0 ] || exit 1
exit 0