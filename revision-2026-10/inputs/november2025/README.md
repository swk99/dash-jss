# November 2025 feature inputs

All 2,438 Parquet files from the author-supplied `btcusdt_2.zip` were checked against `../../manifests/november_file_hashes.json`: 2,438/2,438 match, with no missing or extra files.

For repository distribution, the original 161 MB archive is repackaged into 39 independent ZIP parts. Each part contains complete Parquet files, preserving their exact bytes. These are not concatenation segments: extract all ZIPs into the same directory, or use the standard-library-only helper:

```sh
python revision-2026-10/inputs/november2025/extract_and_verify.py
```

The helper checks each ZIP and all 2,438 original file hashes before extraction, then writes to `extracted/`. Existing different files are never overwritten. Use `extracted/flush_*.parquet` as the experiment data glob.

`verification.json` records part sizes and hashes, the original ZIP SHA-256, and the file verification result. No new latency measurement was performed during publication. This dataset is distinct from the 75 original controlled-replication inputs.
