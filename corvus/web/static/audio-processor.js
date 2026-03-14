/**
 * AudioWorklet processor that captures PCM audio, downsamples to 16 kHz,
 * and sends float32 chunks to the main thread via MessagePort.
 *
 * Register with: audioContext.audioWorklet.addModule('/audio-processor.js')
 * Create with:   new AudioWorkletNode(ctx, 'pcm-capture')
 */

const TARGET_RATE = 16000;

class PCMCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = [];
        this._active = false;

        this.port.onmessage = (e) => {
            if (e.data.command === 'start') {
                this._active = true;
                this._buffer = [];
            } else if (e.data.command === 'stop') {
                this._active = false;
            }
        };
    }

    process(inputs) {
        if (!this._active) return true;

        const input = inputs[0];
        if (!input || !input[0]) return true;

        // input[0] is Float32Array at the AudioContext's sample rate
        const samples = input[0];

        // Downsample from native rate to 16 kHz
        const ratio = sampleRate / TARGET_RATE;
        const outputLen = Math.floor(samples.length / ratio);
        const output = new Float32Array(outputLen);

        for (let i = 0; i < outputLen; i++) {
            // Simple point sampling — sufficient for speech
            const srcIdx = Math.floor(i * ratio);
            output[i] = samples[Math.min(srcIdx, samples.length - 1)];
        }

        if (output.length > 0) {
            this.port.postMessage({ audio: output.buffer }, [output.buffer]);
        }

        return true;
    }
}

registerProcessor('pcm-capture', PCMCaptureProcessor);
