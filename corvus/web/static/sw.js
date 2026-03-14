/* Corvus PWA — service worker (static asset caching only) */

const CACHE_NAME = "corvus-v4";
const STATIC_ASSETS = [
    "/",
    "/index.html",
    "/review.html",
    "/voice.html",
    "/style.css",
    "/app.js",
    "/review.js",
    "/voice.js",
    "/audio-processor.js",
    "/manifest.json",
];

self.addEventListener("install", function (event) {
    event.waitUntil(
        caches.open(CACHE_NAME).then(function (cache) {
            return cache.addAll(STATIC_ASSETS);
        })
    );
    self.skipWaiting();
});

self.addEventListener("activate", function (event) {
    event.waitUntil(
        caches.keys().then(function (names) {
            return Promise.all(
                names
                    .filter(function (name) { return name !== CACHE_NAME; })
                    .map(function (name) { return caches.delete(name); })
            );
        })
    );
    self.clients.claim();
});

self.addEventListener("fetch", function (event) {
    var url = new URL(event.request.url);

    // Never cache API calls — always go to network
    if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/ws/")) {
        return;
    }

    // Static assets: cache-first, fallback to network
    event.respondWith(
        caches.match(event.request).then(function (cached) {
            return cached || fetch(event.request).then(function (response) {
                // Cache new static assets on the fly
                if (response.ok) {
                    var clone = response.clone();
                    caches.open(CACHE_NAME).then(function (cache) {
                        cache.put(event.request, clone);
                    });
                }
                return response;
            });
        })
    );
});
