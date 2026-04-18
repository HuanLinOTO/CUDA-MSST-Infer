# CudaInfer Server Mode Design

## Goal

Add an embedded HTTP server mode to `cudasep_infer` so users can open a browser, upload an audio file, choose a `.csm` model, and run a quick inference test without writing custom scripts.

## Chosen Approach

Use the existing executable as a dual-mode binary:

- CLI mode keeps the current `--model --input --output` workflow.
- Server mode is enabled with `--serve` and scans a model directory for `.csm` files.
- The server is intentionally simple and single-process, with one request handled at a time.
- The runtime keeps at most one loaded model cached to avoid wasting GPU memory.

This is smaller and easier to ship than adding a separate web backend.

## HTTP Surface

- `GET /` serves a lightweight upload form.
- `GET /healthz` returns a basic JSON health response.
- `GET /api/models` returns the scanned model list.
- `POST /api/infer` accepts multipart form uploads with:
  - `audio`: uploaded audio file
  - `model`: model path relative to `--model-dir`
  - `stem`: single stem index to export
  - `overlap`: optional overlap override

The response is a downloadable WAV file for the requested stem.

## Runtime Design

- Shared inference logic is moved out of `main.cpp` into reusable helpers.
- The reusable layer owns model loading, chunking, inference execution, stem naming, and output writing.
- HTTP mode writes the uploaded file to a temporary path, reuses the normal audio loader, runs inference, encodes the selected output stem to WAV bytes, and returns it directly.

## Safety and Constraints

- Default bind address is `127.0.0.1` to avoid accidental exposure.
- Users can override host and port explicitly.
- Upload size is capped with `--max-upload-mb`.
- Model selection is restricted to files under `--model-dir`.
- The first version only returns one stem per request; `-1` / multi-file archive output is deferred.

## Testing Plan

- Rebuild the binary with CMake.
- Smoke test CLI mode to ensure existing inference still works.
- Start server mode locally and verify:
  - form page loads
  - model list endpoint returns expected entries
  - uploaded audio returns a WAV download
  - invalid uploads or model paths return readable errors
