// 共享前端工具：选项化 select + 公共 fetch/SSE
async function jget(url, opts={}) {
  const r = await fetch(url, opts); const j = await r.json().catch(()=>({}));
  return j;
}

function el(tag, cls, txt) { const e = document.createElement(tag); if(cls)e.className=cls; if(txt)e.textContent=txt; return e; }
function clear(node){ while(node.firstChild) node.removeChild(node.firstChild); }

function kindChip(kind) {
  const map = { "v2.1":["v21","v2.1"], "v3.0":["v30","v3.0"], "not_dataset":["nodata","不可用"] };
  const [cls,label] = map[kind] || ["nodata", kind||"-"];
  return label === "不可用" ? `<span class="chip nodata">不可用</span>` : `<span class="chip ${cls}">${label}</span>`;
}
function doneBadges(done) {
  if (!done || !done.length) return '<span class="muted">未处理</span>';
  return done.map(s => `<span class="kid ok">${s}</span>`).join("");
}

// SSE 订阅：把日志追加到 logEl，done 时回调
function subscribe(logEl, taskId, onDone) {
  if (!(window.EventSource && taskId)) return;
  logEl.textContent = "";
  const es = new EventSource(`/api/stream/${taskId}`);
  es.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    logEl.textContent += d.line + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  };
  es.addEventListener("done", (ev) => {
    const d = JSON.parse(ev.data);
    es.close();
    logEl.textContent += `\n[完成 rc=${d.rc}]\n`; logEl.scrollTop = logEl.scrollHeight;
    if (onDone) onDone(d);
  });
  es.onerror = () => {};
}