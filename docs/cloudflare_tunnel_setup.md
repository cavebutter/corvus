# Cloudflare Tunnel Setup for Corvus PWA

Exposes `corvus serve` (FastAPI on port 8095) to the internet via an existing Cloudflare Tunnel, accessible at `corvus.jayco.dev`. Cloudflare Access provides an additional auth layer (email OTP) in front of the API key auth built into Corvus.

## Architecture

```
Phone (PWA)
  → corvus.jayco.dev (Cloudflare Edge, HTTPS termination)
  → Cloudflare Access (email OTP, 30-day session)
  → Existing tunnel on 192.168.0.100 (Unraid, Docker)
  → 192.168.0.92:8095 (Corvus FastAPI, HTTP)
```

- HTTPS is terminated at Cloudflare's edge. Traffic between Cloudflare and the tunnel is encrypted. Traffic from the tunnel to Corvus is plain HTTP on the LAN.
- Two auth layers: Cloudflare Access (blocks unauthenticated traffic before it reaches the server) + Corvus API key (protects API endpoints).

## Network

| Machine | LAN IP | Role |
|---------|--------|------|
| Dev machine (Pop!_OS) | 192.168.0.92 | Runs `corvus serve` on port 8095 |
| Unraid NAS | 192.168.0.100 | Runs `cloudflared` tunnel in Docker |

## Prerequisites

- Cloudflare account with `jayco.dev` domain
- Existing `cloudflared` tunnel (`homelab-unraid`) running in Docker on 192.168.0.100
- Corvus running on 192.168.0.92:8095 (`corvus serve`)
- Cloudflare Zero Trust dashboard access

## Step 1: Add Published Application Route

1. Go to **Cloudflare Zero Trust** → **Networks** → **Connectors**
2. Click your tunnel (`homelab-unraid`) → **Published application routes** tab
3. Add a new route:
   - **Subdomain:** `corvus`
   - **Domain:** `jayco.dev`
   - **Path:** (leave empty)
   - **Service:** `http://192.168.0.92:8095`
4. Save

This automatically creates a DNS record. Do **not** manually create a CNAME record — the published route handles it.

Corvus should now be reachable at `https://corvus.jayco.dev/api/health` (returns `{"status": "ok"}`). The API endpoints will return 403 without a valid API key, which is expected.

## Step 2: Configure Cloudflare Access Policy

This adds an email OTP login page in front of Corvus. Only your email can access it.

1. Go to **Cloudflare Zero Trust** → **Access** → **Applications** → **Add an application**
2. Select **Self-hosted**
3. Configure:
   - **Application name:** `Corvus`
   - **Session duration:** `30 days`
   - **Subdomain:** `corvus`
   - **Domain:** `jayco.dev`
4. Add a policy:
   - **Policy name:** `Owner`
   - **Action:** `Allow`
   - **Include rule:** Emails — enter your email address
5. Save

Now visiting `corvus.jayco.dev` will show a Cloudflare login page. Enter your email, receive a one-time PIN, enter it. The session cookie lasts 30 days.

### WebSocket Bypass

Cloudflare Access may interfere with WebSocket upgrades on `/ws/voice` if the browser doesn't send the Access cookie during the upgrade. In practice, the cookie is sent automatically since the PWA is loaded from the same origin. If voice connections fail after enabling Access, add a bypass rule:

1. In the Corvus application settings → **Add a policy**
2. **Policy name:** `WebSocket bypass`
3. **Action:** `Bypass`
4. **Include rule:** Path — `/ws/voice`

Only add this if needed — test without it first, as the cookie-based auth should work transparently.

## Step 3: Verify

1. **From your phone on cellular (not WiFi):** open `https://corvus.jayco.dev`
2. You should see the Cloudflare Access login page
3. Enter your email, receive OTP, enter it
4. Dashboard should load (will prompt for API key on first visit)
5. Test voice page — tap to talk, verify audio round-trip works

## Step 4: Install as PWA

1. In Safari (iOS) or Chrome (Android), visit `https://corvus.jayco.dev`
2. Tap **Share** → **Add to Home Screen** (iOS) or **Install app** (Android)
3. The app opens in standalone mode (no browser chrome)

## CORS Configuration

The FastAPI app already has CORS middleware. If you see CORS errors, add `https://corvus.jayco.dev` to the allowed origins in `corvus/web/app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://corvus.jayco.dev"],
    ...
)
```

Currently the app allows all origins (`["*"]`), which works but can be tightened once the tunnel is live.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| 502 Bad Gateway | Corvus not running | Start `corvus serve` on 192.168.0.92 |
| 403 Forbidden (after Access login) | Missing/wrong API key | Enter API key in PWA prompt |
| WebSocket connection fails | WebSockets not enabled on tunnel | Enable WebSockets in tunnel hostname settings |
| Voice works on LAN but not via tunnel | Cloudflare Access blocking WS upgrade | Add WebSocket bypass policy (see above) |
| "Service worker registration failed" | Mixed content or wrong scope | Ensure HTTPS — Cloudflare handles this automatically |
| OTP email not arriving | Check spam folder | Also verify the email matches the Access policy |

## Running Corvus as a Service

To keep Corvus running after logout, create a systemd user service:

```bash
# ~/.config/systemd/user/corvus.service
[Unit]
Description=Corvus Web Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/mnt/hdd/PycharmProjects/corvus
ExecStart=/home/jayco/virtual-envs/corvus/.venv/bin/corvus serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable corvus.service
systemctl --user start corvus.service
loginctl enable-linger jayco  # keeps user services running after logout
```
