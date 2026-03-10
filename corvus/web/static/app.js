/* Corvus PWA — main application script */

(function () {
    "use strict";

    const statusDot = document.getElementById("connection-status");

    async function checkHealth() {
        try {
            const res = await fetch("/api/health");
            if (res.ok) {
                statusDot.className = "connected";
                statusDot.title = "Connected";
            } else {
                statusDot.className = "error";
                statusDot.title = "Server error";
            }
        } catch {
            statusDot.className = "error";
            statusDot.title = "Disconnected";
        }
    }

    // Register service worker
    if ("serviceWorker" in navigator) {
        navigator.serviceWorker.register("/sw.js").catch(function (err) {
            console.warn("SW registration failed:", err);
        });
    }

    // Initial health check, then poll every 30s
    checkHealth();
    setInterval(checkHealth, 30000);
})();
