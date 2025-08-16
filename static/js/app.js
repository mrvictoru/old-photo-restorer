const historyListEl = document.getElementById("historyList");
const deleteSelectedBtn = document.getElementById("deleteSelected");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadPreview = document.getElementById("uploadPreview");
const previewImg = document.getElementById("previewImg");

const positivePrompt = document.getElementById("positivePrompt");
const guidanceScale = document.getElementById("guidanceScale");
const strength = document.getElementById("strength");
const steps = document.getElementById("steps");
const seed = document.getElementById("seed");

const runBtn = document.getElementById("runBtn");
const rerunBtn = document.getElementById("rerunBtn");
const statusText = document.getElementById("statusText");

const idleState = document.getElementById("idleState");
const runningState = document.getElementById("runningState");
const errorState = document.getElementById("errorState");
const resultPreview = document.getElementById("resultPreview");
const resultImg = document.getElementById("resultImg");
const downloadLink = document.getElementById("downloadLink");

let state = {
  items: [],
  selectedId: null,
  pollTimer: null,
};

function fmtDate(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Accept": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data.detail) msg = data.detail;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

function renderHistory() {
  historyListEl.innerHTML = "";
  state.items.forEach(item => {
    const li = document.createElement("li");
    li.className = "history-item";
    li.dataset.id = item.id;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.addEventListener("click", e => e.stopPropagation());

    const img = document.createElement("img");
    img.className = "thumb";
    img.src = item.thumb_url || item.input_url;
    img.alt = "thumb";

    const meta = document.createElement("div");
    meta.className = "item-meta";
    const title = document.createElement("div");
    title.className = "item-title";
    title.textContent = item.original_filename || item.id.slice(0, 8);
    const sub = document.createElement("div");
    sub.className = "item-sub";
    sub.textContent = `Uploaded: ${fmtDate(item.uploaded_at)}${item.status === "done" ? " · Done" : item.status === "running" ? " · Running" : ""}`;

    meta.appendChild(title);
    meta.appendChild(sub);

    li.appendChild(checkbox);
    li.appendChild(img);
    li.appendChild(meta);

    li.addEventListener("click", () => selectItem(item.id));
    historyListEl.appendChild(li);
  });
}

async function loadHistory() {
  const data = await fetchJSON("/api/history");
  state.items = data.items || [];
  renderHistory();
}

function getSelectedCheckboxIds() {
  const ids = [];
  historyListEl.querySelectorAll(".history-item").forEach(li => {
    const cb = li.querySelector('input[type="checkbox"]');
    if (cb && cb.checked) ids.push(li.dataset.id);
  });
  return ids;
}

async function deleteSelected() {
  const ids = getSelectedCheckboxIds();
  if (ids.length === 0) return;
  if (!confirm(`Delete ${ids.length} item(s)? This cannot be undone.`)) return;
  await fetchJSON("/api/history", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  if (state.selectedId && ids.includes(state.selectedId)) {
    clearSelectionUI();
  }
  await loadHistory();
}

function clearSelectionUI() {
  state.selectedId = null;
  previewImg.src = "";
  uploadPreview.classList.add("hidden");
  resultImg.src = "";
  resultPreview.classList.add("hidden");
  idleState.classList.remove("hidden");
  runningState.classList.add("hidden");
  errorState.classList.add("hidden");
  errorState.textContent = "";
  runBtn.disabled = true;
  rerunBtn.disabled = true;
  statusText.textContent = "";
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function populateSettingsFromItem(item) {
  if (item.prompts) {
    positivePrompt.value = item.prompts.positive || "";
  }
  if (item.advanced) {
    guidanceScale.value = item.advanced.guidance_scale ?? 7.5;
    strength.value = item.advanced.strength ?? 0.8;
    steps.value = item.advanced.steps ?? 30;
    seed.value = item.advanced.seed ?? "";
  }
}

function updateStatusUI(item) {
  statusText.textContent = `Status: ${item.status}`;
  if (item.status === "running") {
    idleState.classList.add("hidden");
    runningState.classList.remove("hidden");
    errorState.classList.add("hidden");
    resultPreview.classList.add("hidden");
    runBtn.disabled = true;
    rerunBtn.disabled = true;
  } else if (item.status === "done") {
    idleState.classList.add("hidden");
    runningState.classList.add("hidden");
    errorState.classList.add("hidden");
    if (item.result_url) {
      // Add cache-busting to result image URL
      const cacheBustedUrl = item.result_url + "?t=" + Date.now();
      resultImg.src = cacheBustedUrl;
      downloadLink.href = item.result_url; // Keep original URL for download
      resultPreview.classList.remove("hidden");
    }
    runBtn.disabled = false;
    rerunBtn.disabled = false;
  } else if (item.status === "error") {
    idleState.classList.add("hidden");
    runningState.classList.add("hidden");
    errorState.textContent = item.error || "Unknown error";
    errorState.classList.remove("hidden");
    resultPreview.classList.add("hidden");
    runBtn.disabled = false;
    rerunBtn.disabled = false;
  } else {
    idleState.classList.remove("hidden");
    runningState.classList.add("hidden");
    errorState.classList.add("hidden");
    resultPreview.classList.add("hidden");
    runBtn.disabled = false;
    rerunBtn.disabled = true;
  }
}

async function selectItem(id) {
  const item = state.items.find(x => x.id === id);
  if (!item) return;
  state.selectedId = id;
  if (item.input_url) {
    previewImg.src = item.input_url;
    uploadPreview.classList.remove("hidden");
  } else {
    uploadPreview.classList.add("hidden");
  }
  populateSettingsFromItem(item);
  updateStatusUI(item);

  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const s = await fetchJSON(`/api/status/${id}`);
      const idx = state.items.findIndex(x => x.id === id);
      if (idx >= 0) {
        state.items[idx].status = s.status;
        state.items[idx].error = s.error;
        state.items[idx].result_url = s.result_url;
      }
      if (id === state.selectedId) {
        updateStatusUI(state.items[idx]);
      }
      if (s.status === "done" || s.status === "error" || s.status === "idle") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        loadHistory();
      }
    } catch (e) {
      console.warn("Polling failed:", e);
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }, 1500);
}

async function handleUpload(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetchJSON("/api/upload", { method: "POST", body: form });
  await loadHistory();
  if (res.item && res.item.id) {
    await selectItem(res.item.id);
  }
}

dropzone.addEventListener("dragover", e => {
  e.preventDefault();
  dropzone.classList.add("drag");
});
dropzone.addEventListener("dragleave", e => {
  dropzone.classList.remove("drag");
});
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  const file = e.dataTransfer.files?.[0];
  if (file) handleUpload(file);
});
fileInput.addEventListener("change", e => {
  const file = e.target.files?.[0];
  if (file) handleUpload(file);
});

deleteSelectedBtn.addEventListener("click", deleteSelected);

async function runCurrent() {
  if (!state.selectedId) return;
  const payload = {
    id: state.selectedId,
    prompts: {
      positive: positivePrompt.value || "",
    },
    advanced: {
      guidance_scale: parseFloat(guidanceScale.value) || 7.5,
      strength: parseFloat(strength.value) || 0.8,
      steps: parseInt(steps.value, 10) || 30,
      seed: seed.value === "" ? null : parseInt(seed.value, 10),
    }
  };
  await fetchJSON("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const idx = state.items.findIndex(x => x.id === state.selectedId);
  if (idx >= 0) {
    state.items[idx].status = "running";
    updateStatusUI(state.items[idx]);
  }
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const s = await fetchJSON(`/api/status/${state.selectedId}`);
      const idx = state.items.findIndex(x => x.id === state.selectedId);
      if (idx >= 0) {
        state.items[idx].status = s.status;
        state.items[idx].error = s.error;
        state.items[idx].result_url = s.result_url;
      }
      updateStatusUI(state.items[idx]);
      if (s.status === "done" || s.status === "error") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        loadHistory();
      }
    } catch (e) {
      console.warn("Polling failed:", e);
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }, 1500);
}

runBtn.addEventListener("click", runCurrent);
rerunBtn.addEventListener("click", runCurrent);

loadHistory();
