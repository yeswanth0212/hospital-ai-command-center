// =============================================
// Smart Hospital AI — Clinical Decision Engine
// =============================================

const _origin = window.location.origin;
const API = (_origin.includes("file:") || window.location.port !== "8000") ? "http://localhost:8000" : _origin;
let charts = {};
let isRunning = false;
let agentMode = "llm";

let simState = {
    patients: [],
    icu_available: 0,
    general_available: 0,
    survival_rate: 0,
    efficiency_score: 0,
    current_step: 0,
    history: { reward: [], icu: [], gen: [], sev: [1, 1, 1] }
};

// ---- TRIAGE HELPERS ----
function getTriageLevel(sev) {
    if (sev <= 3) return { label: "Easy",   cls: "badge-easy",   color: "#00D4A8" };
    if (sev <= 7) return { label: "Medium", cls: "badge-medium", color: "#FB923C" };
    return              { label: "High",   cls: "badge-high",   color: "#F87171" };
}

// ---- PATIENT LIST (LEFT SIDEBAR) ----
function renderPatients(patients) {
    const list = document.getElementById("patient-list");
    list.innerHTML = "";

    const waiting = patients
        .filter(p => p.status === "WAITING")
        .sort((a, b) => b.severity - a.severity);

    if (!waiting.length) {
        list.innerHTML = '<div style="color:var(--text-dim);font-size:0.8rem;text-align:center;padding:2rem;">No patients in queue</div>';
        return;
    }

    waiting.forEach(p => {
        const t = getTriageLevel(p.severity);
        const div = document.createElement("div");
        div.className = "patient-row";
        div.id = `row-${p.id}`;
        div.innerHTML = `
            <div class="patient-row-top">
                <span class="patient-name">${p.name}</span>
                <span class="tag ${t.cls}">${t.label}</span>
            </div>
            <div class="patient-cond">${p.condition} &nbsp;·&nbsp; Wait: ${p.waiting_time || 0} steps</div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span class="tiny-label">Severity</span>
                <span class="patient-sev" style="color:${t.color}">${p.severity}/10</span>
            </div>
            <div class="sev-bar-bg">
                <div class="sev-bar" style="width:${p.severity * 10}%;background:${t.color}"></div>
            </div>`;
        list.appendChild(div);
    });
}

// ---- DECISION PANEL (CENTER - THE MAIN FOCUS) ----
function renderDecision(suggestion) {
    const name  = suggestion.patient_name || "SYSTEM";
    const act   = (suggestion.action_type || "WAIT").toUpperCase();
    const conf  = Math.round((suggestion.confidence || 0) * 100);
    const whyP  = suggestion.why_patient || "No patient rationale provided.";
    const whyB  = suggestion.why_bed     || "No bed rationale provided.";

    // Patient name + avatar
    document.getElementById("sel-patient-name").textContent = name;
    document.getElementById("avatar-initial").textContent = name === "SYSTEM" ? "⚕" : name[0].toUpperCase();

    // Patient meta chips (severity + condition) — find from state
    const p = simState.patients.find(x => x.id === suggestion.patient_id);
    const metaEl = document.getElementById("sel-patient-meta");
    if (p) {
        const t = getTriageLevel(p.severity);
        metaEl.innerHTML = `
            <span class="meta-chip">Severity: ${p.severity}/10</span>
            <span class="meta-chip">${p.condition}</span>
            <span class="meta-chip tag ${t.cls}" style="margin:0">${t.label}</span>
            <span class="meta-chip">Wait: ${p.waiting_time || 0} steps</span>`;
    } else {
        metaEl.innerHTML = "";
    }

    // Action badge
    const actionEl = document.getElementById("action-badge");
    actionEl.textContent = act;
    actionEl.className = `action-badge ${act === "ICU" ? "action-icu" : act === "GENERAL" ? "action-general" : "action-wait"}`;

    // WHY patient (blue box)
    animateText("why-patient-text", whyP);

    // WHY bed (purple box)
    animateText("why-bed-text", whyB);

    // Confidence
    document.getElementById("conf-bar").style.width = `${conf}%`;
    document.getElementById("conf-pct").textContent = `${conf}%`;
    document.getElementById("conf-badge").textContent = `${conf}% Confidence`;

    // Flash the decision card
    const card = document.querySelector(".decision-card");
    card.style.borderColor = act === "ICU" ? "rgba(248,113,113,0.4)" : act === "GENERAL" ? "rgba(59,130,246,0.4)" : "rgba(0,212,168,0.15)";
    setTimeout(() => { card.style.borderColor = "rgba(0, 212, 168, 0.15)"; }, 1500);

    // Animate selected patient row in sidebar
    if (suggestion.patient_id) {
        const row = document.getElementById(`row-${suggestion.patient_id}`);
        if (row) {
            row.style.background = "rgba(0, 212, 168, 0.08)";
            row.style.borderColor = "rgba(0, 212, 168, 0.3)";
            setTimeout(() => {
                row.style.background = "";
                row.style.borderColor = "";
            }, 1000);
        }
    }
}

// Smooth text animation for WHY boxes
function animateText(elId, text) {
    const el = document.getElementById(elId);
    el.style.opacity = "0";
    setTimeout(() => {
        el.textContent = text;
        el.style.transition = "opacity 0.4s";
        el.style.opacity = "1";
    }, 200);
}

// ---- DECISION LOG (RIGHT SIDEBAR) ----
function addLogEntry(step, suggestion, reward) {
    const log = document.getElementById("decision-log");
    const div = document.createElement("div");
    div.className = "log-entry";

    const rewardVal = typeof reward === "object" ? reward.value : reward;
    const rewardFmt = rewardVal >= 0 ? `+${rewardVal.toFixed(1)}` : rewardVal.toFixed(1);
    const rewardColor = rewardVal >= 0 ? "var(--accent)" : "var(--danger)";

    div.innerHTML = `
        <div class="log-step">Step #${step}</div>
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div class="log-patient">${suggestion.action_type} → ${suggestion.patient_name || "System"}</div>
            <div class="log-reward" style="color:${rewardColor}">${rewardFmt}</div>
        </div>
        <div class="log-why">${(suggestion.why_patient || "").slice(0, 80)}${suggestion.why_patient?.length > 80 ? "..." : ""}</div>`;

    log.prepend(div);
    while (log.children.length > 12) log.lastChild.remove();
}

// ---- METRICS ----
function renderMetrics(obs) {
    document.getElementById("survival-rate").textContent = `${obs.survival_rate || 0}%`;
    document.getElementById("efficiency-score").textContent = `${Math.round(obs.efficiency_score || 0)}%`;
    document.getElementById("icu-avail").textContent = obs.icu_available ?? "—";
    document.getElementById("gen-avail").textContent = obs.general_available ?? "—";

    // Bed availability bars
    const ICU_TOTAL = 5, GEN_TOTAL = 15;
    const icuAvail = obs.icu_available ?? 0;
    const genAvail = obs.general_available ?? 0;

    document.getElementById("icu-count").textContent = `${icuAvail} / ${ICU_TOTAL} available`;
    document.getElementById("gen-count").textContent = `${genAvail} / ${GEN_TOTAL} available`;

    // Bar fills as percentage of total
    document.getElementById("icu-bar").style.width = `${(icuAvail / ICU_TOTAL) * 100}%`;
    document.getElementById("gen-bar").style.width = `${(genAvail / GEN_TOTAL) * 100}%`;
}

// ---- STATE UPDATE ----
function applyState(obs) {
    simState.patients = obs.patients || [];
    simState.icu_available = obs.icu_available;
    simState.general_available = obs.general_available;
    simState.survival_rate = obs.survival_rate;
    simState.efficiency_score = obs.efficiency_score;
    simState.current_step = obs.current_step;

    simState.history.reward.push(obs.cumulative_reward || 0);
    simState.history.icu.push(5 - (obs.icu_available || 0));
    simState.history.gen.push(15 - (obs.general_available || 0));

    const counts = [0, 0, 0];
    simState.patients.forEach(p => {
        if (p.severity <= 3) counts[0]++;
        else if (p.severity <= 7) counts[1]++;
        else counts[2]++;
    });
    simState.history.sev = counts;

    if (simState.history.reward.length > 20) {
        simState.history.reward.shift();
        simState.history.icu.shift();
        simState.history.gen.shift();
    }

    renderPatients(simState.patients);
    renderMetrics(obs);
    updateCharts();
}

// ---- API CALLS ----
async function fetchState() {
    try {
        const r = await fetch(`${API}/state`);
        const obs = await r.json();
        applyState(obs);
    } catch {}
}

async function runAIStep() {
    if (isRunning) return;
    isRunning = true;
    document.getElementById("start-btn").textContent = "⏳ Thinking...";

    try {
        // 1. Get AI suggestion
        const sr = await fetch(`${API}/suggest?mode=${agentMode}`, { method: "POST" });
        const suggestion = await sr.json();

        // 2. Display the WHY immediately
        renderDecision(suggestion);

        // 3. Execute the step
        const action = {
            patient_id: suggestion.patient_id,
            action_type: suggestion.action_type || "WAIT"
        };
        const er = await fetch(`${API}/step`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(action)
        });
        const result = await er.json();

        // 4. Update state + log
        applyState(result.observation);
        addLogEntry(result.observation.current_step, suggestion, result.reward);

    } catch (e) {
        console.error("Step error:", e);
    } finally {
        isRunning = false;
        document.getElementById("start-btn").textContent = "▶ Autonomous Start";
    }
}

async function autoRunSimulation() {
    for (let i = 0; i < 15; i++) {
        await runAIStep();
        await new Promise(r => setTimeout(r, 900));
    }
}

async function resetSim() {
    await fetch(`${API}/reset`, { method: "POST" });
    document.getElementById("decision-log").innerHTML = "";
    document.getElementById("why-patient-text").textContent = "Waiting for AI to analyze the queue...";
    document.getElementById("why-bed-text").textContent = "Waiting for bed type selection...";
    document.getElementById("sel-patient-name").textContent = "— Awaiting Decision —";
    document.getElementById("avatar-initial").textContent = "?";
    document.getElementById("sel-patient-meta").innerHTML = "";
    simState.history = { reward: [], icu: [], gen: [], sev: [1, 1, 1] };
    await fetchState();
}

function setMode(mode) {
    agentMode = mode;
    document.getElementById("mode-llm").classList.toggle("active", mode === "llm");
    document.getElementById("mode-heuristic").classList.toggle("active", mode === "heuristic");
}

async function changeScenario() {
    const sc = document.getElementById("scenario-select").value;
    await fetch(`${API}/scenario?scenario=${sc}`, { method: "POST" });
    await fetchState();
}

// ---- CHARTS ----
function initCharts() {
    if (typeof Chart === "undefined") { setTimeout(initCharts, 400); return; }

    const baseOpts = {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { x: { display: false }, y: { display: false } }
    };

    charts.reward = new Chart(document.getElementById("rewardChart"), {
        type: "line",
        data: {
            labels: Array.from({ length: 20 }, (_, i) => i),
            datasets: [{
                data: [],
                borderColor: "#00D4A8",
                backgroundColor: "rgba(0,212,168,0.06)",
                fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2
            }]
        },
        options: baseOpts
    });

    charts.usage = new Chart(document.getElementById("usageChart"), {
        type: "bar",
        data: {
            labels: Array.from({ length: 20 }, (_, i) => i),
            datasets: [
                { data: [], backgroundColor: "#F871712A", borderColor: "#F87171", borderWidth: 1 },
                { data: [], backgroundColor: "#3B82F62A", borderColor: "#3B82F6", borderWidth: 1 }
            ]
        },
        options: { ...baseOpts, scales: { x: { display: false, stacked: true }, y: { display: false, stacked: true } } }
    });

    charts.severity = new Chart(document.getElementById("severityChart"), {
        type: "doughnut",
        data: {
            labels: ["Easy", "Medium", "High"],
            datasets: [{
                data: [1, 1, 1],
                backgroundColor: ["#00D4A8", "#FB923C", "#F87171"],
                borderWidth: 0
            }]
        },
        options: { ...baseOpts, cutout: "65%" }
    });
}

function updateCharts() {
    if (!charts.reward) return;
    charts.reward.data.datasets[0].data = simState.history.reward;
    charts.reward.update("none");
    charts.usage.data.datasets[0].data = simState.history.icu;
    charts.usage.data.datasets[1].data = simState.history.gen;
    charts.usage.update("none");
    charts.severity.data.datasets[0].data = simState.history.sev;
    charts.severity.update("none");
}

// ---- INIT ----
window.addEventListener("load", () => {
    initCharts();
    fetchState();
    lucide.createIcons();
});

// Auto-refresh every 5s
setInterval(fetchState, 5000);
