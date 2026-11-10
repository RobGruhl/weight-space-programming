# Research references

Source: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), authors of *LLMs can understand and write neural network weights*.
Pinned revision: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.

The public export links to upstream rather than redistributing its logs, prompts or data. The [license notice](UPSTREAM-LICENSE) is preserved for attribution. To reproduce the patch analyses, obtain the unmodified public source into the ignored `references/upstream` directory:

```sh
git clone --filter=blob:none --no-checkout https://github.com/RikiyaT/lm_understand-logs.git references/upstream
git -C references/upstream checkout 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f
```

The local code reads `exp8_math/runs/p50c/edit/edit.npz` and `manifest.json`; it does not execute upstream code. The patch archive SHA-256 is `e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7`.

Read the upstream README and license before use. The upstream evaluation harness refers to components not supplied there, so this is not a turnkey recreation of its original benchmark. Model files and any quoted benchmark content retain their respective licenses.
