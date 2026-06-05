# HRM-Text CUDA

A native Rust/CUDA inference runner for the `sapientinc/HRM-Text-1B` checkpoint.

## Important Model Limitation

HRM-Text-1B is a pre-alignment base model, not a chat or instruction-tuned assistant. This app defaults to treating each message as an independent prompt because that produces substantially more reliable output. Conversation context is available as an experimental option, but it can reduce quality.

## Run

The default interface is a local web chat:

```powershell
cargo run --release -- .\hf_cache
```

Then open `http://127.0.0.1:8765`.

Use a different port:

```powershell
cargo run --release -- --port 8765 .\hf_cache
```

Use the terminal fallback:

```powershell
cargo run --release -- --terminal .\hf_cache
```

Pass a Hugging Face repository instead of a local model directory to download missing model files into `.\hf_cache`.

## Build Requirements

- Rust stable
- NVIDIA GPU with CUDA support
- CUDA toolkit with `nvcc`
- Visual Studio C++ Build Tools on Windows

If `nvcc` cannot find `cl.exe`, run the build from a Visual Studio Developer PowerShell or Developer Command Prompt.

## Interface Controls

- **Reasoning** uses the checkpoint's `synth,cot` composite condition.
- **Direct** uses the checkpoint's direct-answer condition.
- **Conversation context** includes a structured, capped transcript and is off by default.
- **Web research** searches public web pages, extracts bounded readable evidence, and shows the sources under the response.
- **Quick research** runs one focused search; **Deep research** runs several query variants and reads more sources.
- **Temperature 0** uses greedy decoding and is the most stable setting.
- **Top-k**, **top-p**, and **repetition penalty** only affect sampled generation.

Web research uses public DuckDuckGo results with Yahoo and Brave Search fallbacks and does not require an API key. Successful reports are briefly cached to reduce provider throttling. Retrieved page text is treated as untrusted evidence, private-network targets are blocked, and source downloads are capped. HRM first compresses the retrieved evidence into question-specific cited notes, then performs a second pass to write the final answer. Since HRM-Text-1B is a base model, bracketed citations in its prose may still be inconsistent; the attached source cards are the reliable audit trail.
