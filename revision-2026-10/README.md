# DASH JSS revision evidence

Published from the author-supplied revision packages on 22 October 2026.

## Available material

- `inputs/original75/`: all 75 controlled-replication Parquet feature files. Every file's SHA-256 matches `manifests/original_expected_sha256.json`.
- `breadth_runs/cloud_original75/`, `breadth_runs/laptop_original75/`, `breadth_runs/laptop_original75_run2/`, and `breadth_runs/shared_november2025/`: four supplied breadth runs, including per-request Parquet records and available metadata. Keep environments and occasions separate.
- `scripts/` and `evidence_recomputed/`: supplied revision analysis/plotting scripts and derived tables. `scripts/extension/` holds the additional experiment driver and protocol.
- `manuscript/` and `supplement/`: supplied v4 documents, sources, and figures. The previously missing figure is `supplement/figures/figS_update_cost.pdf`.
- Original implementation, model artifacts and archived controlled-study evidence: https://github.com/swk99/dash-bound-state/tree/53353760db37921a8da5e231169d9dcec3027cff (referenced, not duplicated here).
- `SHA256SUMS.json`: integrity inventory of the supplied revision files (excluding this README and the separately pinned frozen-source snapshot).

## Outstanding items and document status

The November 2025 input archive (`btcusdt_2.zip`, 2,438 feature files) is **not included**. Its file hashes and the shared-host run outputs are included. The original75 input is a different dataset and is not a substitute.

The supplied manuscript and supplement are archived as received, not a newly compiled or fully reconciled submission. Their availability statements and some section references may be stale. In particular, the original 75 inputs are now available here; the November archive is not. Update the final manuscript to this actual repository URL and availability status before submission.

The author confirmed that the cloud-container breadth run was executed by Claude in its sandbox, and the other breadth runs, including the November shared-host run, were executed by the author. Runtime paths alone do not establish who executed a run.

No new latency experiment was executed for this upload. There is no author-controlled cloud rerun or new external current-feature-vector-only Redis baseline in this publication. The latter remains optional proposed work.

The standalone Java/DuckDB rule-processing harness belongs to a different study and is intentionally not included as evidence for this Python/Redis ML manuscript. Local filesystem inventory and superseded internal editorial reports are also excluded.

## Reanalysis

Use the frozen implementation's documentation and the supplied scripts. For example, from this directory:

```sh
git clone https://github.com/swk99/dash-bound-state.git frozen-source
git -C frozen-source checkout 53353760db37921a8da5e231169d9dcec3027cff
python scripts/reanalyse.py frozen-source recomputed-local
```

See `scripts/extension/experiment_protocol.txt` for the additional experiment procedure. Its historical host descriptions document the original supplied run, not a new execution. Do not claim exact November input reproduction until that archive is supplied and verified.
