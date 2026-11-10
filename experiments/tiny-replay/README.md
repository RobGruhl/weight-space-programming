# Local p50c implementation smoke replay

This one-off experiment verifies and applies the published internal-MLP p50c edit to the pinned Qwen3.5-0.8B text model. It is not a reproduction of MATH-500, GSM8K, IFEval, discovery of the edit, or the email experiment's proposed output-head intervention. All inputs are synthetic arithmetic problems, not personal email. Generation is offline; downloads were public model files and official PyPI dependencies. No paid model APIs or rented compute are used.

## Reproducibility and limits

- Upstream: RikiyaT/lm_understand-logs at `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`; retain its notices in `references/upstream/LICENSE`. Our adapter does not execute the copied upstream scripts.
- Base: Qwen/Qwen3.5-0.8B at `2fc06364715b967f1860aea9cf38778875588b17`. The upstream file `model.safetensors-00001-of-00001.safetensors` is locally named `model.safetensors`, with SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. It includes an unused vision tower; only text tensors are loaded. The source model license is preserved alongside the checkpoint.
- Hardware: Apple M4 Pro, 24 GB RAM. The initial CPU-only attempt used four PyTorch threads; the completed separately labeled run used Apple MPS after a BF16 support check. No quantization. Text checkpoint preserves 284 BF16 and 36 FP32 tensors; the two edited tensors are BF16.
- Runtime: Python 3.13.15, torch 2.15.0, transformers 5.22.2. The full environment is pinned in `requirements.lock.txt`. PyTorch reference attention/convolution kernels are used, not CUDA/vLLM or optimized linear-attention kernels.
- Four fixed synthetic prompts, each baseline then p50c, thinking enabled, seed 0 reset per prompt, temperature 0.6, top-p 0.95, top-k 20, generated-token presence penalty 1.5. One sample each. Exact prompts, rendered token IDs, hashes and outputs are in private run artifacts.
- Maximum 256 new tokens and approximately 60 seconds per generation, checked between tokens. Stop early for four repeated identical 16-token blocks. The outer process watchdog was 720 seconds for CPU and 600 seconds for MPS. These are not upstream's 32,768-token benchmark conditions. Seed matching does not make CPU or MPS sampling equivalent to vLLM.
- A cutoff is not evidence of looping. Missing boxed answers in truncated generations are recorded as extraction failures, not evidence that the model could not solve the problem. The exact-string boxed-answer checker is only for these four integer answers, not a benchmark grader. No statistical capability claim is supported by this sample.

## Integrity

The adapter verifies the downloaded model file, checks each edited tensor's base BF16 hash, applies only sparse uint16 bit replacements from `edit.npz` with pickle disabled, verifies exact reconstructed hashes and changed-entry counts, then hashes every model state tensor to ensure only the two selected tensors changed. Baseline weights are never overwritten on disk. The two edits change 65,260 and 64,928 entries, totaling 130,188. An independent raw-file-offset check also matched both base/edited manifest hashes.

No discovery training or auxiliary inference is involved. Upstream's evaluation script imports the absent private `weightenv` package and is not presented as runnable here.

## Local files

The one-off environment and all model/run artifacts are under ignored `private/tiny-replay/`. `run-manifest.json` records source files, code hash, configuration, runtime and deviations; `patch-reconstruction.json` is the raw-byte audit; `integrity.json` records live model state verification; `results.json` stores each output; `run.log` captures diagnostics. `replay.py` is the standalone adapter. The isolated manual environment was used for this bounded one-off script. Do not install globally or alter another project's environment.

Two setup attempts stopped before any completed generation: an overly strict all-BF16 check found the source FP32 tensors, and a meta-device initialization left non-persistent rotary buffers uninitialized. The current adapter preserves source precision and initializes buffers normally before loading every checkpoint parameter strictly. These were adapter issues, not model or patch failures.

See `RESULTS.md` for the completed MPS results and preserved partial CPU attempt. The CPU variant can occupy the CPU for about eight minutes; the MPS variant completed generation in under one minute. Review the resource budget and existing processes first. Preserve prior artifacts under a distinct run directory before reuse; the present script overwrites its own run reports, not model weights or personal data.

## Grading schema correction after execution

The original parser wrote `correct: false` for missing boxed answers, which could be mistaken for a capability score. Both the completed MPS results and partial CPU results now label truncated generations `grading_status: "truncated_ungraded"` and `correct: null`. `exact_box_match` retains the original parser boolean, not a capability judgment. Original bytes are preserved as `results.raw.json`, and logs remain raw execution records. Both adapters now use this explicit grading schema for future runs; no inference was rerun.

Each run directory preserves its actual executed script as `adapter.executed.py`; its SHA-256 still matches that run’s unchanged manifest. The later revised source hashes and before/after result hashes are recorded in `private/tiny-replay/grading-schema-update.json`. Current source hashes therefore intentionally differ from the executed-code hashes.
