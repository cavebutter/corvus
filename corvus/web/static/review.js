/* Corvus PWA — review interface */

(function () {
    "use strict";

    var KEY_STORAGE = "corvus_api_key";
    var statusDot = document.getElementById("connection-status");

    function getApiKey() {
        return localStorage.getItem(KEY_STORAGE) || "";
    }

    function headers() {
        return { "X-API-Key": getApiKey(), "Content-Type": "application/json" };
    }

    // ── Auth check ──────────────────────────────────────────────────

    if (!getApiKey()) {
        window.location.href = "/";
        return;
    }

    // ── Health ──────────────────────────────────────────────────────

    fetch("/api/health").then(function (r) {
        statusDot.className = r.ok ? "connected" : "error";
    }).catch(function () { statusDot.className = "error"; });

    // ── Tabs ────────────────────────────────────────────────────────

    var tabs = document.querySelectorAll(".review-tab");
    tabs.forEach(function (tab) {
        tab.addEventListener("click", function () {
            tabs.forEach(function (t) { t.classList.remove("active"); });
            tab.classList.add("active");
            document.querySelectorAll(".review-panel").forEach(function (p) {
                p.style.display = "none";
            });
            document.getElementById("tab-" + tab.dataset.tab).style.display = "";
        });
    });

    // ── API helpers ─────────────────────────────────────────────────

    async function apiFetch(path) {
        var res = await fetch(path, { headers: headers() });
        if (res.status === 401) { window.location.href = "/"; return null; }
        if (!res.ok) return null;
        return res.json();
    }

    async function apiPost(path, body) {
        var res = await fetch(path, {
            method: "POST",
            headers: headers(),
            body: JSON.stringify(body || {}),
        });
        return res;
    }

    // ── Confidence bar ──────────────────────────────────────────────

    function confidenceHtml(value) {
        var pct = Math.round(value * 100);
        var color = pct >= 90 ? "#22c55e" : pct >= 70 ? "#f59e0b" : "#ef4444";
        return '<div class="confidence-bar">' +
            '<div class="confidence-fill" style="width:' + pct + '%;background:' + color + '"></div>' +
            '<span class="confidence-text">' + pct + '%</span></div>';
    }

    // ── Format action ───────────────────────────────────────────────

    function formatEmailAction(action) {
        if (!action) return "—";
        var type = action.action_type || "keep";
        if (type === "move" && action.target_folder) {
            return "Move → " + action.target_folder;
        }
        return type.charAt(0).toUpperCase() + type.slice(1);
    }

    // ── Render email items ──────────────────────────────────────────

    function renderEmailItem(item) {
        var task = item.task;
        var cls = task.classification;

        var card = document.createElement("article");
        card.className = "review-card";
        card.id = "item-" + item.id;

        card.innerHTML =
            '<div class="review-card-header">' +
                '<span class="review-from">' + escHtml(task.from_address) + '</span>' +
                '<span class="review-category badge-' + cls.category + '">' + cls.category.replace(/_/g, " ") + '</span>' +
            '</div>' +
            '<div class="review-subject">' + escHtml(task.subject) + '</div>' +
            (cls.summary ? '<div class="review-summary">' + escHtml(cls.summary) + '</div>' : '') +
            '<div class="review-meta">' +
                '<span>Action: <strong>' + formatEmailAction(task.proposed_action) + '</strong></span>' +
                confidenceHtml(task.overall_confidence) +
            '</div>' +
            '<div class="review-reasoning">' + escHtml(cls.reasoning) + '</div>' +
            '<div class="review-actions">' +
                '<button class="btn-approve" data-id="' + item.id + '" data-type="email">Approve</button>' +
                '<button class="btn-reject secondary" data-id="' + item.id + '" data-type="email">Reject</button>' +
            '</div>';

        return card;
    }

    // ── Render document items ───────────────────────────────────────

    function renderDocItem(item) {
        var task = item.task;
        var result = task.result;
        var tags = result.suggested_tags.map(function (t) { return t.tag_name; }).join(", ");

        var card = document.createElement("article");
        card.className = "review-card";
        card.id = "item-" + item.id;

        card.innerHTML =
            '<div class="review-card-header">' +
                '<span class="review-doc-title">' + escHtml(task.document_title) + '</span>' +
                '<span class="review-doc-id">id=' + task.document_id + '</span>' +
            '</div>' +
            '<div class="review-meta">' +
                '<span>Tags: <strong>' + escHtml(tags) + '</strong></span>' +
                confidenceHtml(task.overall_confidence) +
            '</div>' +
            (result.suggested_correspondent ?
                '<div class="review-meta"><span>Correspondent: <strong>' + escHtml(result.suggested_correspondent) + '</strong></span></div>' : '') +
            (result.suggested_document_type ?
                '<div class="review-meta"><span>Type: <strong>' + escHtml(result.suggested_document_type) + '</strong></span></div>' : '') +
            '<div class="review-reasoning">' + escHtml(result.reasoning) + '</div>' +
            (task.content_snippet ?
                '<details class="review-snippet"><summary>Content snippet</summary><p>' + escHtml(task.content_snippet.substring(0, 300)) + '</p></details>' : '') +
            '<div class="review-actions">' +
                '<button class="btn-approve" data-id="' + item.id + '" data-type="documents">Approve</button>' +
                '<button class="btn-reject secondary" data-id="' + item.id + '" data-type="documents">Reject</button>' +
            '</div>';

        return card;
    }

    // ── HTML escaping ───────────────────────────────────────────────

    function escHtml(str) {
        if (!str) return "";
        var div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Action handlers ─────────────────────────────────────────────

    function attachActions(container) {
        container.addEventListener("click", async function (e) {
            var btn = e.target.closest(".btn-approve, .btn-reject");
            if (!btn) return;

            var id = btn.dataset.id;
            var type = btn.dataset.type;
            var action = btn.classList.contains("btn-approve") ? "approve" : "reject";
            var card = document.getElementById("item-" + id);

            // Optimistic UI
            card.classList.add("review-card-resolved");
            btn.disabled = true;

            var res = await apiPost("/api/review/" + type + "/" + id + "/" + action, {});

            if (res && res.ok) {
                card.classList.add("review-card-done");
                setTimeout(function () {
                    card.remove();
                    updateBadges();
                    checkEmpty();
                }, 400);
            } else {
                // Revert on error
                card.classList.remove("review-card-resolved");
                btn.disabled = false;
                var msg = "Action failed";
                try {
                    var body = await res.json();
                    msg = body.detail || msg;
                } catch (ignored) {}
                alert(msg);
            }
        });
    }

    // ── Badge counts ────────────────────────────────────────────────

    function updateBadges() {
        var emailCount = document.querySelectorAll("#email-list .review-card:not(.review-card-done)").length;
        var docCount = document.querySelectorAll("#doc-list .review-card:not(.review-card-done)").length;
        document.getElementById("email-badge").textContent = emailCount;
        document.getElementById("doc-badge").textContent = docCount;
    }

    function checkEmpty() {
        var emailList = document.getElementById("email-list");
        var docList = document.getElementById("doc-list");
        document.getElementById("email-empty").style.display =
            emailList.children.length === 0 ? "" : "none";
        document.getElementById("doc-empty").style.display =
            docList.children.length === 0 ? "" : "none";
    }

    // ── Load data ───────────────────────────────────────────────────

    async function loadReview() {
        var emailItems = await apiFetch("/api/review/email");
        var docItems = await apiFetch("/api/review/documents");

        var emailList = document.getElementById("email-list");
        var docList = document.getElementById("doc-list");
        emailList.innerHTML = "";
        docList.innerHTML = "";

        if (emailItems) {
            emailItems.forEach(function (item) {
                emailList.appendChild(renderEmailItem(item));
            });
        }

        if (docItems) {
            docItems.forEach(function (item) {
                docList.appendChild(renderDocItem(item));
            });
        }

        updateBadges();
        checkEmpty();
    }

    // ── Init ────────────────────────────────────────────────────────

    attachActions(document.getElementById("email-list"));
    attachActions(document.getElementById("doc-list"));
    loadReview();
})();
