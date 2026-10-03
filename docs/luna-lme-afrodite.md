# Archived full-dream Luna diagnostic

The pinned `siwc_lme_diagnostic_v12.py` runner and `siwc_lme_launch_v11.py` launcher remain in this staging tree to audit the earlier full-dream repair. The prepared capsule `/home/atta/.hymem-siwc-lme-diagnostic-luna-v12-20261003b` is source-bound and can be checked without model calls:

```bash
RUNTIME=/home/atta/.hymem-siwc-runtime-v1/bin/python
LAUNCHER=/home/atta/HyMem-Luna-LME/tools/diagnostics/siwc_lme_launch_v11.py
CAPSULE=/home/atta/.hymem-siwc-lme-diagnostic-luna-v12-20261003b
"$RUNTIME" -I -B "$LAUNCHER" --preflight-root "$CAPSULE"
```

The full-dream run was stopped after the first row proved too slow for the requested 500-row evaluation. Its output is diagnostic, with `canonical_r9_artifact=false` and `official_model_score=false`. The selected 500-row path is the [Luna no-dream guide](/home/atta/HyMem-Luna-LME/docs/luna-lme-nodream-afrodite.md); it uses SIWC OAuth for the reader, local judge, and optional message rerank, and its receipts prohibit API fallback.
