# FireRedTTS3

FireRedTTS3-Base is a zero-shot voice-cloning TTS model built on continuous
speech representations. A Qwen3-1.7B backbone consumes text plus the patchified
reference latents and emits one hidden state per 160 ms audio patch; a DiT
flow-matching head turns each hidden state into four RedAE latent frames, and a
sigmoid stop head ends the utterance. RedAE decodes the latents to 24 kHz audio.

sglang-omni serves this as a four-stage pipeline:

```text
preprocessing -> reference_encode -> tts_engine (SGLang) -> vocoder
   text front end   RedAE encode      Qwen3 backbone +      RedAE decode
   + tokenizer      + CAM++ speaker   DiT flow head         + prompt trim
```

The backbone runs inside SGLang (KV cache, continuous batching, abort handling);
the patch encoder, DiT flow head, and stop head run in float32 as upstream does.

## Serve

```bash
hf download FireRedTeam/FireRedTTS3 --local-dir pretrained_models/FireRedTTS3

sgl-omni serve \
  --model-path pretrained_models/FireRedTTS3 \
  --config examples/configs/fireredtts3.yaml \
  --allowed-local-media-path docs/_static/audio \
  --port 8000
```

`--model-path` points at the repository root, not at `fireredtts3_base/`: the
reference encoder needs `redae/`, `campp/`, and `text_tokenizer/` from the same
checkout. The root ships no `config.json`, so the architecture is resolved from
`fireredtts3_base/config.json`.

Only `FireRedTTS3-Base` (zero-shot cloning) is served today. The Instruct
checkpoint adds voice design and speech editing with a different prompt
protocol and is not wired up.

## Request

Every request needs a reference clip **and its transcript**: the model does
in-context learning over `<|language|><|sot|>{ref_text}{text}<|eot|>`. A
transcript that does not match the clip makes the model re-read the reference
text before your input, which shows up as extra leading speech.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTeam/FireRedTTS3",
    "input": "Open infrastructure makes speech serving reproducible for everyone.",
    "voice": "alloy",
    "ref_audio": "docs/_static/audio/female-voice.wav",
    "ref_text": "By repeating what students say, teachers can demonstrate that they are listening.",
    "response_format": "wav",
    "language": "English",
    "seed": 1234
  }' --output speech.wav
```

`references[0].audio_path` + `references[0].text` and uploaded voices
(`POST /v1/audio/voices`) work as well.

### Language

`language` accepts the 24 language tags and 21 Chinese dialect tags the model
was trained on, for example `English`, `Chinese`, `Cantonese`, `Japanese`,
`ZH_Sichuan`, `ZH_Shanghai`. Omit it (or pass `auto`) to detect Chinese,
Japanese, or English from the text. The base model relies on an explicit tag, so
pass one whenever you know it — especially for dialects, which detection never
returns.

Written-form numbers, dates, and units are normalized for Chinese, English, and
Cantonese when `wetext` is installed. Without it the raw text reaches the model;
the server logs a warning at startup.

### Sampling knobs

| Field | Default | Meaning |
|-------|---------|---------|
| `n_timesteps` | 10 | Flow-matching ODE steps per patch. Lower is faster, rougher |
| `inference_cfg` | 2.0 | Classifier-free guidance on the DiT. `0` disables the CFG branch |
| `stop_threshold` | 0.5 | Stop-head probability that ends the utterance |
| `min_gen_steps` | 6 | Patches generated before the stop head is honoured |
| `max_gen_steps` | 400 | Patch budget; 400 patches is 64 s of audio |
| `seed` | unset | Seeds a per-request generator for the flow-head noise |

With a `seed`, a request is reproducible regardless of the batch it lands in;
the noise comes from a per-request generator, not the global RNG.

## Limits

- **One segment per request.** Upstream splits long text into sentences and
  cross-fades the segments; this pipeline synthesizes a single segment and
  rejects input longer than 300 characters. Split long text client-side.
- **No streaming.** RedAE decodes the reference and generated latents together
  with full attention, so the waveform is produced once generation ends.
- **No decode CUDA graph.** Each decode step consumes a DiT-produced embedding,
  so `disable_cuda_graph` stays on; the backbone decodes eagerly.
- **TP1 only.**

## Verification

- Bit-exact parity against upstream `FireRedTTS3BaseCore.generate` for the
  rewritten recurrence (prefill layout, condition history, stop gate, flow head,
  per-request RNG): identical latents over 54 AR steps, `max |Δ| = 0`.
- Served output on an H20, 4-way concurrency, `whisper-medium` transcription:
  WER 0 for English (`docs/_static/audio/female-voice.wav`,
  `male-voice.wav`), CER 0 for Chinese after simplified/traditional
  normalization. CAM++ cosine similarity to the reference speaker 0.82-0.87,
  versus 0.05-0.16 against the other speaker.
- Putting the flow head in eval mode with gradients off (needed to stop the
  rollout retaining an autograd graph) leaves output bit-identical: the same
  4-request concurrency set is byte-for-byte equal before and after, as expected
  since the DiT is built with `dropout=0.0`.
- Same-length output as upstream for the same request and seed (4.48 s), which
  means the stop head fires on the same step even though SGLang runs the
  backbone in bf16 while upstream uses fp32 weights under bf16 autocast.

## Performance

One H20, `examples/configs/fireredtts3.yaml` (`max_running_requests: 8`), one
English sentence per request producing 5.12 s of audio. `rtf` is
latency / generated audio seconds; `realtime` is total generated audio over wall
clock.

| concurrency | p50 latency | p99 | rtf | throughput | aggregate |
|-------------|-------------|-----|-----|------------|-----------|
| 1 | 4.63 s | 4.83 s | 0.89 | 0.22 req/s | 1.1x realtime |
| 2 | 4.48 s | 4.65 s | 0.89 | 0.45 req/s | 2.3x realtime |
| 4 | 4.56 s | 4.65 s | 0.89 | 0.87 req/s | 4.5x realtime |
| 8 | 4.39 s | 4.58 s | 0.92 | 1.80 req/s | 8.6x realtime |
| 16 | 8.04 s | 8.65 s | 1.60 | 1.93 req/s | 8.8x realtime |

Latency is flat up to 8 concurrent requests, so the batch is free: SGLang runs
the backbone and the flow head batches its DiT rollout across requests. At 16
the extra requests queue behind `max_running_requests: 8`, which doubles latency
without adding throughput; raise that limit (and `mem_fraction_static`) to go
further.

For reference, upstream's single-process implementation on the same GPU and
request takes 3.99 s (rtf 0.78, 0.25 req/s). One served request is ~16% slower
because it crosses the four-stage pipeline over IPC, but the served pipeline
reaches **7.2x** upstream's throughput once requests overlap.
