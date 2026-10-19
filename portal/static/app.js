"use strict";
// Meme Studio front end: vanilla JS, talks only to this server's /api/*.

const $ = (id) => document.getElementById(id);
const state = {
  file: null, presets: [], status: null, jobs: [],
  opts: { preset: "dramatic_zoom", quality: "draft", duration: "5", aspect: "auto", audio: true },
  viewing: null,
};
const ACTIVE = ["queued", "running"];

// ------------------------------------------------------------------ helpers
async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: "same-origin", ...opts });
  if (res.status === 401 && path !== "/api/login") { showLogin(); throw new Error("Enter the password first"); }
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new Error(body?.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `HTTP ${res.status}`);
  return body;
}
const jsonPost = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

function toast(msg, bad = false) {
  const t = document.createElement("div");
  t.className = "toast" + (bad ? " bad" : "");
  t.textContent = msg;
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), bad ? 6000 : 3000);
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
const money = (x) => "$" + (x ?? 0).toFixed(x && x < 0.1 ? 3 : 2);
function elapsed(job) {
  const s = Math.max(0, Math.round(Date.now() / 1000 - (job.created || 0)));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}
function savePrefs() { try { localStorage.setItem("memeStudio", JSON.stringify(state.opts)); } catch {} }
function loadPrefs() { try { Object.assign(state.opts, JSON.parse(localStorage.getItem("memeStudio") || "{}")); } catch {} }

// ------------------------------------------------------------------ create panel
function renderPresets() {
  const box = $("presets");
  box.replaceChildren();
  for (const p of state.presets) {
    const b = el("button", "preset");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(p.id === state.opts.preset));
    b.style.setProperty("--c1", p.colors[0]);
    b.style.setProperty("--c2", p.colors[1]);
    b.append(el("span", "em", p.emoji), el("span", "nm", p.name), el("span", "tg", p.tag));
    b.onclick = () => { state.opts.preset = p.id; savePrefs(); renderPresets(); syncForm(); };
    box.appendChild(b);
  }
}

function currentPreset() { return state.presets.find((p) => p.id === state.opts.preset) || state.presets[0]; }

function estimate() {
  const s = state.status;
  if (!s) return null;
  let usd = s.est_usd_per_clip_s[state.opts.quality] * Number(state.opts.duration);
  const cold = ["asleep", "waking", "no_gpu", "unknown"].includes(s.gpu.state) && !s.mock;
  if (cold) usd += s.cold_start_usd;
  return { usd: s.mock ? 0 : usd, cold };
}

function syncForm() {
  const p = currentPreset();
  if (!p) return;
  $("lineField").hidden = !p.needs_line;
  $("customField").hidden = !p.custom;
  if (p.needs_line) { $("lineLabel").textContent = p.line_label; $("line").placeholder = p.line_placeholder || ""; }
  $("presetHint").textContent = `${p.emoji} ${p.name}`;
  document.querySelectorAll(".seg").forEach((seg) => {
    seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(state.opts[seg.dataset.name] === b.dataset.v)));
  });
  $("soundBtn").setAttribute("aria-pressed", String(state.opts.audio));
  $("soundBtn").textContent = state.opts.audio ? "🔊 Sound" : "🔇 Sound";

  const est = estimate();
  $("goCost").textContent = est ? (state.status.mock ? "mock · free" : `~${money(est.usd)}`) : "";
  let note = "", ok = !!state.file;
  if (!state.file) note = "Add a photo to start.";
  else if (p.needs_line && !$("line").value.trim()) { ok = false; note = "Type what they should say."; }
  else if (p.custom && $("custom").value.trim().length < 20) { ok = false; note = "Describe the motion and sound in a sentence or two."; }
  else if (state.status && state.status.active_jobs >= state.status.max_active) { ok = false; note = "A few memes are cooking already — hang on."; }
  else if (est?.cold) note = "GPU is asleep: the first meme takes 2–5 extra minutes to wake it. The next ones are fast.";
  else if (state.status?.gpu.state === "warm") note = "GPU is warm — this one starts right away.";
  if (!state.status?.captions && ($("capTop").value || $("capBottom").value)) note = "ffmpeg isn't installed on the server, so captions won't be burned in.";
  $("goBtn").disabled = !ok;
  $("goNote").textContent = note;
  $("capTopPreview").textContent = $("capTop").value;
  $("capBottomPreview").textContent = $("capBottom").value;
}

function setFile(f) {
  if (!f || !f.type.startsWith("image/")) { toast("That's not an image", true); return; }
  state.file = f;
  const url = URL.createObjectURL(f);
  $("previewImg").onload = () => URL.revokeObjectURL(url);
  $("previewImg").src = url;
  $("preview").hidden = false;
  $("dropEmpty").hidden = true;
  syncForm();
}

async function generate() {
  if ($("goBtn").disabled) return;
  const p = currentPreset();
  const fd = new FormData();
  fd.append("image", state.file);
  fd.append("preset", p.id);
  fd.append("subject", $("subject").value.trim());
  fd.append("line", $("line").value.trim());
  fd.append("custom", $("custom").value.trim());
  fd.append("caption_top", $("capTop").value);
  fd.append("caption_bottom", $("capBottom").value);
  fd.append("quality", state.opts.quality);
  fd.append("duration", state.opts.duration);
  fd.append("aspect", state.opts.aspect);
  fd.append("audio", String(state.opts.audio));
  const seed = $("seed").value.trim();
  if (seed) fd.append("seed", seed);
  $("goBtn").disabled = true;
  $("goText").textContent = "Sending…";
  try {
    const job = await api("/api/generate", { method: "POST", body: fd });
    state.jobs.unshift(job);
    renderFeed();
    toast(`${p.emoji} ${p.name} is cooking`);
    refreshStatus();
  } catch (e) { toast(e.message, true); }
  finally { $("goText").textContent = "Generate"; syncForm(); }
}

// ------------------------------------------------------------------ feed
function card(job) {
  const c = el("article", "card");
  c.dataset.id = job.id;
  const media = el("div", "media");
  media.style.aspectRatio = `${job.width} / ${job.height}`;
  media.tabIndex = 0;

  if (job.status === "completed") {
    const v = el("video");
    Object.assign(v, { src: job.video_url, muted: true, loop: true, playsInline: true, autoplay: true, preload: "metadata" });
    media.appendChild(v);
    const ov = el("div", "overlay");
    const snd = el("button", "icon-btn", "🔇");
    snd.title = "Sound";
    snd.onclick = (e) => { e.stopPropagation(); v.muted = !v.muted; snd.textContent = v.muted ? "🔇" : "🔊"; if (!v.muted) v.play(); };
    const dl = el("a", "icon-btn", "⬇");
    dl.href = job.video_url.replace(/\?.*/, "") + "?dl=1";
    dl.title = "Download";
    dl.onclick = (e) => e.stopPropagation();
    const rm = el("button", "icon-btn", "🎲");
    rm.title = "New take (new seed)";
    rm.onclick = (e) => { e.stopPropagation(); remix(job, {}); };
    ov.append(snd, rm, dl);
    media.appendChild(ov);
    media.onclick = () => openViewer(job.id);
    media.onkeydown = (e) => { if (e.key === "Enter") openViewer(job.id); };
  } else {
    const img = el("img", "busy-bg");
    img.src = job.poster;
    img.alt = "";
    media.appendChild(img);
    if (ACTIVE.includes(job.status)) media.appendChild(el("div", "shimmer"));
    const st = el("div", "state" + (job.status === "failed" ? " failed" : ""));
    if (ACTIVE.includes(job.status)) {
      st.append(el("div", "big", job.emoji), el("div", "stage", job.stage || "Working"), el("div", "elapsed", elapsed(job)));
      const cancel = el("button", "mini", "Cancel");
      cancel.onclick = async (e) => {
        e.stopPropagation();
        try { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); toast("Cancelling…"); poll(); } catch (err) { toast(err.message, true); }
      };
      st.appendChild(cancel);
    } else {
      st.append(el("div", "big", job.status === "failed" ? "💀" : "🚫"),
        el("div", "stage", job.status === "failed" ? "Failed" : "Cancelled"));
      if (job.error) st.appendChild(el("div", "errmsg", job.error));
      const row = el("div");
      const retry = el("button", "mini", "Retry");
      retry.onclick = (e) => { e.stopPropagation(); remix(job, { keep_seed: true }); };
      const del = el("button", "mini", "Remove");
      del.onclick = (e) => { e.stopPropagation(); removeJob(job.id); };
      row.append(retry, " ", del);
      st.appendChild(row);
    }
    media.appendChild(st);
  }
  const foot = el("div", "card-foot");
  foot.append(el("span", "t", `${job.emoji} ${job.caption_top || job.caption_bottom || job.preset_name}`));
  const meta = [job.quality === "hd" ? "HD" : "Draft", `${job.duration}s`];
  if (job.status === "completed" && job.cost_usd != null) meta.push(money(job.cost_usd));
  foot.append(el("span", "m", meta.join(" · ")));
  c.append(media, foot);
  return c;
}

let renderedOnce = false;
function renderFeed() {
  const feed = $("feed");
  // keep finished cards (and their playing videos) untouched; rebuild only what changed
  const existing = new Map([...feed.children].map((n) => [n.dataset.id, n]));
  const nodes = state.jobs.map((j) => {
    const key = `${j.status}|${j.stage}|${j.video_url}|${j.caption_top}|${j.caption_bottom}`;
    const old = existing.get(j.id);
    if (old && old.dataset.key === key && !ACTIVE.includes(j.status)) return old;
    const n = card(j);
    n.dataset.key = key;
    if (!old && renderedOnce) n.classList.add("new");          // only fresh cards animate in
    return n;
  });
  feed.replaceChildren(...nodes);
  renderedOnce = true;
  $("empty").hidden = state.jobs.length > 0;
  const done = state.jobs.filter((j) => j.status === "completed").length;
  $("count").textContent = state.jobs.length ? `${done} done` : "";
}

async function remix(job, body) {
  try {
    const nj = await jsonPost(`/api/jobs/${job.id}/remix`, body);
    state.jobs.unshift(nj);
    renderFeed();
    toast(body.quality === "hd" ? "✨ Rendering the HD version" : "🎲 New take is cooking");
    refreshStatus();
  } catch (e) { toast(e.message, true); }
}

async function removeJob(id) {
  try {
    await api(`/api/jobs/${id}`, { method: "DELETE" });
    state.jobs = state.jobs.filter((j) => j.id !== id);
    renderFeed();
  } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ viewer
function openViewer(id) {
  const job = state.jobs.find((j) => j.id === id);
  if (!job) return;
  state.viewing = id;
  $("vVideo").src = job.video_url;
  $("vTitle").textContent = `${job.emoji} ${job.preset_name}`;
  const meta = [job.quality === "hd" ? "HD" : "Draft", `${job.width}×${job.height}`, `${job.duration}s`, `seed ${job.seed}`];
  if (job.cost_usd != null) meta.push(`~${money(job.cost_usd)}`);
  if (job.gpu) meta.push(job.gpu);
  $("vMeta").textContent = meta.join(" · ");
  $("vTop").value = job.caption_top || "";
  $("vBottom").value = job.caption_bottom || "";
  $("vDownload").href = job.video_url.replace(/\?.*/, "") + "?dl=1";
  $("vHd").hidden = job.quality === "hd";
  $("vPrompt").textContent = job.prompt;
  $("viewer").showModal();
  $("vVideo").play().catch(() => {});
}
function closeViewer() { $("vVideo").pause(); $("vVideo").removeAttribute("src"); $("viewer").close(); state.viewing = null; }

// ------------------------------------------------------------------ status & polling
async function refreshStatus() {
  try {
    const s = await api("/api/status");
    state.status = s;
    $("mockBadge").hidden = !s.mock;
    $("gpuChip").dataset.state = s.gpu.state;
    $("gpuLabel").textContent = s.gpu.label;
    $("gpuChip").title = s.gpu.detail || "";
    $("spendChip").textContent = `${money(s.spent_today_usd)} / $${s.budget_usd} today`;
    $("spendChip").classList.toggle("over", s.spent_today_usd >= s.budget_usd);
    if (s.auth_required && !s.authed) showLogin();
    syncForm();
  } catch (e) { /* network blip; next poll retries */ }
}

async function poll() {
  try {
    state.jobs = await api("/api/jobs");
    renderFeed();
  } catch (e) { /* ignored, retried */ }
}

let timer = null;
function schedule() {
  clearTimeout(timer);
  const busy = state.jobs.some((j) => ACTIVE.includes(j.status));
  timer = setTimeout(async () => {
    await Promise.all([poll(), refreshStatus()]);
    schedule();
  }, busy ? 2000 : 10000);
}
// tick elapsed timers on cooking cards between polls
setInterval(() => {
  document.querySelectorAll(".card").forEach((c) => {
    const j = state.jobs.find((x) => x.id === c.dataset.id);
    const e = c.querySelector(".elapsed");
    if (j && e) e.textContent = elapsed(j);
  });
}, 1000);

// ------------------------------------------------------------------ login
function showLogin() { if (!$("login").open) { $("login").showModal(); $("pw").focus(); } }
$("loginForm").onsubmit = async (e) => {
  e.preventDefault();
  try {
    await jsonPost("/api/login", { password: $("pw").value });
    $("login").close();
    $("loginErr").textContent = "";
    await boot();
  } catch (err) { $("loginErr").textContent = err.message; }
};
$("login").addEventListener("cancel", (e) => e.preventDefault());

// ------------------------------------------------------------------ wiring
function wire() {
  const drop = $("drop");
  $("file").onchange = (e) => e.target.files[0] && setFile(e.target.files[0]);
  $("swapBtn").onclick = (e) => { e.preventDefault(); $("file").click(); };
  drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } };
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) setFile(item.getAsFile());
  });

  document.querySelectorAll(".seg").forEach((seg) => seg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    state.opts[seg.dataset.name] = b.dataset.v;
    savePrefs();
    syncForm();
  }));
  $("soundBtn").onclick = () => { state.opts.audio = !state.opts.audio; savePrefs(); syncForm(); };
  ["capTop", "capBottom", "line", "custom", "subject"].forEach((id) => $(id).addEventListener("input", syncForm));
  $("goBtn").onclick = generate;
  document.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") generate(); });

  $("vClose").onclick = closeViewer;
  $("viewer").addEventListener("close", () => { $("vVideo").pause(); state.viewing = null; });
  $("viewer").addEventListener("click", (e) => { if (e.target === $("viewer")) closeViewer(); });
  $("vSaveCaps").onclick = async () => {
    try {
      const j = await jsonPost(`/api/jobs/${state.viewing}/captions`, { caption_top: $("vTop").value, caption_bottom: $("vBottom").value });
      state.jobs = state.jobs.map((x) => (x.id === j.id ? j : x));
      $("vVideo").src = j.video_url;
      $("vVideo").play().catch(() => {});
      renderFeed();
      toast("Captions updated");
    } catch (e) { toast(e.message, true); }
  };
  $("vRemix").onclick = () => { const j = state.jobs.find((x) => x.id === state.viewing); closeViewer(); remix(j, {}); };
  $("vHd").onclick = () => { const j = state.jobs.find((x) => x.id === state.viewing); closeViewer(); remix(j, { quality: "hd", keep_seed: true }); };
  $("vDelete").onclick = () => { const id = state.viewing; closeViewer(); removeJob(id); };
}

async function boot() {
  await refreshStatus();
  if (state.status?.auth_required && !state.status.authed) return;
  try {
    state.presets = await api("/api/presets");
  } catch (e) { toast(e.message, true); return; }
  if (!state.presets.some((p) => p.id === state.opts.preset)) state.opts.preset = state.presets[0].id;
  renderPresets();
  syncForm();
  await poll();
  schedule();
}

loadPrefs();
wire();
boot();
