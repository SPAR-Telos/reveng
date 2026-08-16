# Maze smoke-test package status

## Ready now

- The fixed API design is prepared without making generation calls.
- Eighteen no-key/no-door mazes cover sizes 7, 11, and 15 and generator
  difficulty settings 0.0, 0.6, and 1.0.
- Each condition uses median-path and long-path grids selected from 32
  deterministic candidates. All models and reasoning conditions share the same
  grids.
- The schedule contains 108 resumable trajectories: 54 GPT-OSS-20B, 27
  Gemma-4-31B-it, and 27 Qwen3-32B.
- API records include exact inference name, provider-returned model name,
  system fingerprint when available, decoding parameters, seed, prompt hash,
  token usage, latency, parse status, planner labels, and failure status.
- Local checkpoints are pinned to Hugging Face commit SHAs.
- The local preflight checks layers 8, 15, and 23, reasoning-token alignment,
  tensor shape, finite values, peak VRAM, runtime, and activation bytes.
- Unit tests exercise grid reuse, schedule balance, action parsing, resumability,
  planner behavior, reports, reasoning-span detection, and tensor validation.

## Intentionally not run here

- No paid maze API generation has been performed.
- No model weights have been downloaded for this smoke test.
- No local checkpoint has been loaded, so `local_preflight.md` correctly says
  `Ready for full activation run: NO`.

## External blocker

Together currently serves GPT-OSS-20B and Gemma-4-31B-it under the configured
serverless names. Qwen3-32B requires an account-specific dedicated endpoint
inference name. Supply it during the plan command as documented in
`REMOTE_RUNBOOK.md`; the paid runner refuses to start while the placeholder
remains.

## Required order

1. Create or identify the Qwen3-32B endpoint.
2. Freeze a fresh plan with that endpoint name.
3. Run the non-billable API availability check.
4. Run the API trajectories and inspect `api_summary.md`.
5. Use each model's recommended token cap in the local preflight.
6. Proceed to full activation generation only if both reports pass.
