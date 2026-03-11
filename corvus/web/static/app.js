/* Corvus PWA — main application script */

(function () {
    "use strict";

    var KEY_STORAGE = "corvus_api_key";
    var POLL_INTERVAL = 30000;

    var statusDot = document.getElementById("connection-status");
    var authPrompt = document.getElementById("auth-prompt");
    var dashboard = document.getElementById("dashboard");
    var apiKeyInput = document.getElementById("api-key-input");
    var apiKeySubmit = document.getElementById("api-key-submit");

    function getApiKey() {
        return localStorage.getItem(KEY_STORAGE) || "";
    }

    function setApiKey(key) {
        localStorage.setItem(KEY_STORAGE, key);
    }

    function headers() {
        return { "X-API-Key": getApiKey() };
    }

    // ── Auth ────────────────────────────────────────────────────────

    function showAuth() {
        authPrompt.style.display = "";
        dashboard.style.display = "none";
    }

    function showDashboard() {
        authPrompt.style.display = "none";
        dashboard.style.display = "";
    }

    if (apiKeySubmit) {
        apiKeySubmit.addEventListener("click", function () {
            var key = apiKeyInput.value.trim();
            if (key) {
                setApiKey(key);
                showDashboard();
                refresh();
            }
        });
        apiKeyInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter") apiKeySubmit.click();
        });
    }

    // ── API helpers ─────────────────────────────────────────────────

    async function apiFetch(path) {
        var res = await fetch(path, { headers: headers() });
        if (res.status === 401) {
            showAuth();
            return null;
        }
        if (!res.ok) return null;
        return res.json();
    }

    // ── Health check ────────────────────────────────────────────────

    async function checkHealth() {
        try {
            var res = await fetch("/api/health");
            if (res.ok) {
                statusDot.className = "connected";
                statusDot.title = "Connected";
            } else {
                statusDot.className = "error";
                statusDot.title = "Server error";
            }
        } catch (e) {
            statusDot.className = "error";
            statusDot.title = "Disconnected";
        }
    }

    // ── Status cards ────────────────────────────────────────────────

    function setText(id, value) {
        var el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    function setHighlight(id, value) {
        var el = document.getElementById(id);
        if (!el) return;
        el.textContent = value;
        if (value > 0) {
            el.classList.add("highlight");
        } else {
            el.classList.remove("highlight");
        }
    }

    async function refreshStatus() {
        var data = await apiFetch("/api/status");
        if (!data) return;

        var doc = data.documents;
        var email = data.email;

        setHighlight("doc-pending", doc.pending_review);
        setText("doc-processed", doc.processed_24h);
        setText("doc-reviewed", doc.reviewed_24h);

        setHighlight("email-pending", email.pending_review);
        setText("email-auto", email.auto_applied_24h);
        setText("email-sender-list", email.sender_list_24h);
        setText("email-queued", email.queued_24h);
        setText("email-approved", email.approved_24h);
        setText("email-rejected", email.rejected_24h);
    }

    // ── Activity feed ───────────────────────────────────────────────

    function formatTime(isoString) {
        var d = new Date(isoString);
        return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }

    function formatAction(action) {
        return action.replace(/_/g, " ");
    }

    function renderActivityItem(entry, type) {
        var div = document.createElement("div");
        div.className = "activity-item";

        var time = document.createElement("span");
        time.className = "activity-time";
        time.textContent = formatTime(entry.timestamp);

        var action = document.createElement("span");
        action.className = "activity-action " + entry.action;
        action.textContent = formatAction(entry.action);

        var detail = document.createElement("span");
        detail.className = "activity-detail";

        if (type === "email") {
            detail.textContent = entry.subject || entry.from_address || "";
        } else if (type === "document") {
            detail.textContent = entry.document_title || "";
        } else {
            detail.textContent = entry.file_name || "";
        }

        div.appendChild(time);
        div.appendChild(action);
        div.appendChild(detail);
        return div;
    }

    async function refreshActivity() {
        // Fetch recent entries from all three audit logs
        var results = await Promise.all([
            apiFetch("/api/audit/email?limit=20"),
            apiFetch("/api/audit/documents?limit=10"),
            apiFetch("/api/audit/watchdog?limit=10"),
        ]);

        var emailEntries = results[0] || [];
        var docEntries = results[1] || [];
        var wdEntries = results[2] || [];

        // Tag each with type and merge
        var all = [];
        emailEntries.forEach(function (e) { e._type = "email"; all.push(e); });
        docEntries.forEach(function (e) { e._type = "document"; all.push(e); });
        wdEntries.forEach(function (e) { e._type = "watchdog"; all.push(e); });

        // Sort by timestamp descending
        all.sort(function (a, b) {
            return new Date(b.timestamp) - new Date(a.timestamp);
        });

        // Take the most recent 25
        all = all.slice(0, 25);

        var feed = document.getElementById("activity-feed");
        feed.innerHTML = "";

        if (all.length === 0) {
            feed.innerHTML = '<p class="muted">No recent activity.</p>';
            return;
        }

        all.forEach(function (entry) {
            feed.appendChild(renderActivityItem(entry, entry._type));
        });

        var age = document.getElementById("activity-age");
        if (age) {
            age.textContent = "Updated " + new Date().toLocaleTimeString([], {
                hour: "2-digit", minute: "2-digit"
            });
        }
    }

    // ── Main loop ───────────────────────────────────────────────────

    async function refresh() {
        await checkHealth();
        await refreshStatus();
        await refreshActivity();
    }

    // Register service worker
    if ("serviceWorker" in navigator) {
        navigator.serviceWorker.register("/sw.js").catch(function (err) {
            console.warn("SW registration failed:", err);
        });
    }

    // Boot
    if (getApiKey()) {
        showDashboard();
        refresh();
    } else {
        showAuth();
        checkHealth();
    }

    setInterval(refresh, POLL_INTERVAL);
})();
