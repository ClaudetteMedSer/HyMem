# Luna LME diagnostic progress reader

`luna_lme_diagnostic_progress_v1.py` is a read-only metadata reader, kept
outside the accepted benchmark bundle. On the prepared Linux host, run:

```sh
/usr/bin/python3 -I -B /path/to/luna_lme_diagnostic_progress_v1.py \
  --root /home/atta/.hymem-lme-diagnostic-preflight-oi233cee \
  --receipt-sha256 67ce8fec27a8848612c9ae99f3cbd9cf42da41c31762ed7970eca25604548c2c
```

The reader pins the accepted runner, nine code files, inventory, and all 514
candidate file hashes without importing benchmark code. It reads only the
launch receipt/marker, projected checkpoint, terminal result, and read-only
systemd/cgroup status. It never reads private question rows, prompts, logs, or
stores, and never starts or retries a service. The prepared, unlaunched root
reports `prepared_not_launched` without querying systemd.

`completed_diagnostic_and_clean` requires all four selected questions scored,
reconciled checkpoint/result and known usage, no campaign stop, a structurally
valid canary, successful service exit, and an empty cgroup (including
descendants). It is **not** a canonical LongMemEval score or a claim that the
model matched the canary gold. Strict indexing health, summary degradation,
and canary gold match are reported separately. Stage-specific timing is not
available in the accepted runner and remains explicitly unavailable.
