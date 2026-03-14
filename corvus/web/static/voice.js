/**
 * Corvus Voice — WebSocket voice client.
 *
 * Tap-to-talk: tap to start recording, tap again to stop and send.
 * Audio captured via AudioWorklet at 16 kHz, sent as float32 PCM over WS.
 * TTS audio received as float32 PCM, played via AudioContext.
 */

const API_KEY = localStorage.getItem('corvus_api_key');
if (!API_KEY) {
    window.location.href = '/';
}

// ── State ───────────────────────────────────────────────────────────

let ws = null;
let audioCtx = null;
let workletNode = null;
let mediaStream = null;
let recording = false;
let ttsChunks = [];
let ttsSampleRate = 24000;

const stateEl = document.getElementById('voice-state');
const logEl = document.getElementById('voice-log');
const talkBtn = document.getElementById('talk-btn');
const connEl = document.getElementById('connection-status');
const voiceUI = document.getElementById('voice-ui');

// ── WebSocket ───────────────────────────────────────────────────────

function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/voice?api_key=${encodeURIComponent(API_KEY)}`;

    ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        connEl.className = 'connected';
        connEl.textContent = '●';
    };

    ws.onclose = (e) => {
        connEl.className = 'error';
        connEl.textContent = '●';
        if (e.code === 4001) {
            localStorage.removeItem('corvus_api_key');
            window.location.href = '/';
            return;
        }
        // Reconnect after 3s
        setTimeout(connect, 3000);
    };

    ws.onerror = () => {
        connEl.className = 'error';
    };

    ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
            // Binary TTS audio chunk
            ttsChunks.push(new Float32Array(event.data));
            return;
        }

        const msg = JSON.parse(event.data);
        handleServerMessage(msg);
    };
}

function handleServerMessage(msg) {
    switch (msg.type) {
        case 'ready':
            setState('idle');
            voiceUI.style.display = '';
            break;

        case 'listening':
            setState('recording');
            break;

        case 'transcription':
            if (msg.text) {
                addLogEntry('user', msg.text);
            } else {
                setState('idle', 'No speech detected');
            }
            break;

        case 'thinking':
            setState('thinking');
            break;

        case 'response':
            if (msg.text) {
                addLogEntry('corvus', msg.text);
            }
            break;

        case 'audio_start':
            ttsChunks = [];
            ttsSampleRate = msg.sample_rate || 24000;
            setState('speaking');
            break;

        case 'audio_end':
            playTTSAudio();
            break;

        case 'error':
            addLogEntry('error', msg.message || 'Unknown error');
            setState('idle');
            break;

        case 'pong':
            break;
    }
}

// ── Audio capture ───────────────────────────────────────────────────

async function initAudio() {
    if (audioCtx) return;

    audioCtx = new AudioContext();

    // Load AudioWorklet
    await audioCtx.audioWorklet.addModule('/audio-processor.js');

    // Get microphone
    mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
            channelCount: 1,
            sampleRate: { ideal: 48000 },
            echoCancellation: true,
            noiseSuppression: true,
        }
    });

    const source = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, 'pcm-capture');

    // Forward PCM chunks to WebSocket
    workletNode.port.onmessage = (e) => {
        if (recording && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(new Float32Array(e.data.audio));
        }
    };

    source.connect(workletNode);
    // Don't connect to destination — we don't want mic feedback
}

function startRecording() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    recording = true;
    talkBtn.classList.add('active');
    ws.send(JSON.stringify({ type: 'audio_start' }));
    workletNode.port.postMessage({ command: 'start' });
    setState('recording');
}

function stopRecording() {
    if (!recording) return;

    recording = false;
    talkBtn.classList.remove('active');
    workletNode.port.postMessage({ command: 'stop' });
    ws.send(JSON.stringify({ type: 'audio_end' }));
    setState('processing', 'Transcribing...');
}

// ── TTS playback ────────────────────────────────────────────────────

function playTTSAudio() {
    if (ttsChunks.length === 0) {
        setState('idle');
        return;
    }

    // Concatenate all chunks
    const totalLen = ttsChunks.reduce((sum, c) => sum + c.length, 0);
    const fullAudio = new Float32Array(totalLen);
    let offset = 0;
    for (const chunk of ttsChunks) {
        fullAudio.set(chunk, offset);
        offset += chunk.length;
    }
    ttsChunks = [];

    // Play via AudioContext
    const buffer = audioCtx.createBuffer(1, fullAudio.length, ttsSampleRate);
    buffer.getChannelData(0).set(fullAudio);

    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);
    source.onended = () => setState('idle');
    source.start();
}

// ── UI helpers ──────────────────────────────────────────────────────

function setState(state, message) {
    const labels = {
        idle: 'Tap to speak',
        recording: 'Listening... tap to stop',
        processing: message || 'Processing...',
        thinking: 'Thinking...',
        speaking: 'Speaking...',
    };
    stateEl.textContent = labels[state] || state;
    stateEl.className = `voice-state voice-state-${state}`;

    talkBtn.disabled = (state === 'thinking' || state === 'speaking' || state === 'processing');
}

function addLogEntry(role, text) {
    const div = document.createElement('div');
    div.className = `voice-entry voice-entry-${role}`;

    const label = document.createElement('span');
    label.className = 'voice-entry-label';
    label.textContent = role === 'user' ? 'You' : role === 'corvus' ? 'Corvus' : 'Error';

    const content = document.createElement('span');
    content.className = 'voice-entry-text';
    content.textContent = text;

    div.appendChild(label);
    div.appendChild(content);
    logEl.appendChild(div);
    logEl.scrollTop = logEl.scrollHeight;
}

// ── Event listeners ─────────────────────────────────────────────────

talkBtn.addEventListener('click', async () => {
    // Resume AudioContext on first interaction (browser policy)
    if (!audioCtx) {
        try {
            await initAudio();
        } catch (err) {
            addLogEntry('error', 'Microphone access denied');
            return;
        }
    }
    if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
    }

    if (recording) {
        stopRecording();
    } else {
        startRecording();
    }
});

// ── Keepalive ───────────────────────────────────────────────────────

setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
    }
}, 30000);

// ── Init ────────────────────────────────────────────────────────────

connect();
