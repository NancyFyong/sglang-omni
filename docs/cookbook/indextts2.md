# IndexTTS-2.5

IndexTTS-2.5 is a zero-shot voice-cloning TTS model for Chinese, English,
Japanese, Spanish, and Arabic with emotion control disentangled from timbre. A
GPT-2 backbone predicts discrete semantic ("mel") codes; a semantic codec plus a
flow-matching speech-to-mel decoder and BigVGAN turn them into 22.05 kHz audio.

sglang-omni serves this as a four-stage pipeline:

```text
preprocessing  -> reference_encode        -> tts_engine (SGLang) -> vocoder
text front end    w2v-BERT + CAM++ +         GPT-2 mel-code AR      semantic codec
+ tiktoken        emotion conditioner        (native sampling)      + s2mel + BigVGAN
                  + s2mel prompt
```

The GPT-2 backbone runs inside SGLang, so mel codes are sampled with SGLang's
own sampler and KV cache; every other operator comes from the upstream
`indextts` package.

## Install

The model operators are not published on PyPI, so `indextts` is pinned to a
commit. Its own pins (torch 2.8, numpy 2.2, keras 2.9) would downgrade the
sglang-omni stack, so install it without dependencies — sglang-omni already
declares the extra runtime imports it needs:

```bash
uv pip install --no-deps "indextts @ git+https://github.com/index-tts/index-tts@ee40fa7d"
```

## Serve

```bash
hf download IndexTeam/IndexTTS-2.5 --local-dir checkpoints/IndexTTS-2.5

# w2v-bert-2.0, MaskGCT semantic codec, CAMPPlus, BigVGAN (~5 GB)
python -c "from indextts.utils.model_download import ensure_models_available; \
ensure_models_available('checkpoints/IndexTTS-2.5')"

sgl-omni serve \
  --model-path checkpoints/IndexTTS-2.5 \
  --config examples/configs/indextts2.yaml \
  --allowed-local-media-path docs/_static/audio \
  --port 8000
```

The checkpoint ships `config.yaml` and loose `*.pth` files with no
`config.json` and no `model_type`, so the architecture is resolved by inspecting
the YAML; the engine builder then materializes a config shim with `gpt.pth`
linked into it.

The auxiliary models land in `<checkpoint>/hf_cache/`. Startup fails with a
pointer to the download command if any of them is missing.

## Request

Only reference audio is required — no transcript, unlike FireRedTTS3 or
dots.tts.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "IndexTeam/IndexTTS-2.5",
    "input": "Open infrastructure makes speech serving reproducible for everyone.",
    "voice": "alloy",
    "ref_audio": "docs/_static/audio/female-voice.wav",
    "response_format": "wav",
    "language": "en"
  }' --output speech.wav
```

`references[0].audio_path` and uploaded voices (`POST /v1/audio/voices`) work as
well. Reference clips are truncated to the leading 15 seconds, as upstream does.

### Language

`language` accepts `zh`, `en`, `ja`, `es`, `ar`, and `zhen` (mixed Chinese and
English). Omit it or pass `auto` to detect Japanese, Chinese, Arabic, or English
from the text. Text normalization runs for `zh`/`zhen`/`en` (upstream front end)
and for `ja`/`es` (NeMo); pass `text_normalization: false` to skip it.

Pronunciation control uses upstream's `<word|reading>` syntax, e.g.
`"他在银<行|XING2>里<行|HANG2>走了半天。"` for Pinyin, CMU phonemes for English,
and Kana for Japanese.

### Emotion and duration

| Field | Default | Meaning |
|-------|---------|---------|
| `emotion_audio` | speaker clip | Separate clip that supplies the emotion |
| `emotion_alpha` | 1.0 | Blend between the speaker's own emotion and `emotion_audio` |
| `emotion_vector` | unset | 8 floats: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm |
| `emotion_bias` | true | Apply upstream's per-emotion de-emphasis and the 0.8 sum cap |
| `emotion_random` | false | Sample the emotion prototype instead of matching the speaker |
| `duration_factor` | 1.0 | >1 slows down, <1 speeds up; range [0.5, 2.0] |

### Sampling knobs

| Field | Default | Meaning |
|-------|---------|---------|
| `temperature` | 0.8 | |
| `top_p` | 0.8 | |
| `top_k` | 30 | |
| `repetition_penalty` | 10.0 | CTRL-style penalty applied model-side (see below) |
| `max_mel_tokens` | 1500 | Code budget; the checkpoint allows up to 1815 |
| `diffusion_steps` | 25 | Flow-matching steps in the speech-to-mel decoder |
| `inference_cfg_rate` | 0.7 | Classifier-free guidance in that decoder |

SGLang caps its native `repetition_penalty` at 2.0, while this model ships 10.0
and depends on it to avoid repeated codes. The penalty is therefore applied
inside the model on the mel logits — over the same token set upstream uses (the
prefix placeholder id, the mel start token, and every sampled code) — and
SGLang's sampler runs with 1.0.

## Limits

- **One segment per request.** Upstream splits long text into segments and
  concatenates them with silence; this pipeline synthesizes one segment and
  rejects more than 400 text tokens. Split long text client-side.
- **No beam search.** Upstream's `num_beams=3` default is not available;
  generation is pure sampling.
- **No emotion from text.** Upstream's QwenEmotion path (`use_emo_text`) is not
  wired up; use `emotion_vector` or `emotion_audio`.
- **No streaming.** The speech-to-mel decoder consumes the whole code sequence
  together with the reference prompt.
- **No decode CUDA graph**, and **TP1 only**.

## Verification

Served on an H20 at 4-way concurrency, `whisper-medium` transcription: WER 0 for
English (`docs/_static/audio/female-voice.wav`, `male-voice.wav`) and CER 0 for
Chinese after simplified/traditional normalization (the residual numbers
whisper reports are its own hyphenation and script conversion). CAMPPlus cosine
similarity to the reference speaker 0.65-0.93, versus 0.36-0.46 against the
other speaker.

Against upstream, greedily decoded on both sides for the same request:

| | duration | WER | speaker similarity |
|---|---|---|---|
| upstream | 4.83 s | 0.00 | 0.918 |
| served, 3 runs | 4.75-5.39 s | 0.00-0.25 | 0.914-0.926 |

Note that upstream is **not** reproducible even against itself: three greedy
runs of the same request emitted 136, 123 and 122 codes, because the w2v-BERT
front end is numerically non-deterministic and greedy `argmax` flips near ties.
Code-level parity is therefore not a well-defined target for this model. The
right comparison is whether the port sits inside upstream's own run-to-run
spread, and it does: speaker-embedding cosine is 0.942-0.950 between upstream
and the served output, versus 0.949-0.965 between two served runs of the same
request. (Frame-aligned mel correlation is not usable here — it is 0.09-0.42
even between two served runs, because prosody and timing move every run.)

## Performance

One H20, `examples/configs/indextts2.yaml`, one English sentence per request.
`rtf` is latency / generated audio seconds.

| concurrency | p50 latency | p99 | rtf | throughput | aggregate |
|-------------|-------------|-----|-----|------------|-----------|
| 1 | 2.28 s | 2.49 s | 0.48 | 0.44 req/s | 2.1x realtime |
| 2 | 2.83 s | 3.93 s | 0.61 | 0.65 req/s | 3.2x realtime |
| 4 | 4.03 s | 5.41 s | 0.83 | 0.85 req/s | 4.3x realtime |
| 8 | 7.25 s | 9.86 s | 1.37 | 0.96 req/s | 4.9x realtime |
| 16 | 15.13 s | 20.59 s | 2.92 | 0.90 req/s | 4.4x realtime |

Throughput saturates near 1 req/s from 8 concurrent requests on, and latency
then grows linearly, so **the vocoder is the bottleneck, not the AR stage**: the
GPT backbone batches inside SGLang, but the speech-to-mel decoder runs
`diffusion_steps` flow steps per request and upstream's `cfm.estimator`
allocates its caches with `max_batch_size=1`, so those requests serialize
through `vocoder.max_concurrency` (2 by default). Raising `max_concurrency`
trades memory for a little more overlap; batching the flow decoder would need
upstream changes.

Upstream's single-process implementation on the same GPU and request takes
1.88 s (0.53 req/s), i.e. one served request is ~21% slower because it crosses
the four-stage pipeline, and the served pipeline reaches ~1.8x upstream
throughput under load. That is a much smaller win than FireRedTTS3 sees (7.2x)
for the same reason: here the non-batchable decoder dominates.
