# Reproduction scope and version safeguards

Compact weather fitting CLI/processed datasets, dynamic result-folder Python helpers, and the frozen YC evaluation WTH/templates required by hash validators are included. The installed DSSAT executable and Python environment are external; REPLAY_ENVIRONMENT.json records the current verified CPU replay versions.

deliverables is the active 100K result set. Original five-scenario/daily archives remain under results, and may include old FQ2018 or alternative automatic-irrigation reports. Do not substitute these archive panels for the active 2V02 corrected tables. Historical selection tables are retained as provenance; they have not been silently recomputed using revised endpoints.

YC exact WP_ET completion covers observed-weather 2014-2023 for both eight-seed regimes. Supplemental heldout-WGEN episode outputs are retained in their original form; their WP_ET was not added in this release. Undefined PFP_N when N=0 remains undefined. V01 retains the original partial YC efficiency coverage.

Large CSVs are stored as byte-exact chunks for GitHub transport. Run restore_large_files.py after checkout before workflows that consume those logical source CSV paths; sha256 is checked automatically. Unrelated research, repeated input copies, system executables and large failed caches are not part of the curated release.

The code is a source snapshot of the current workspace; the model/source manifests and release hashes are authoritative. Some extra helper scripts refer to unrelated historical experiments; this package audits the listed 80 policies, not every exploratory script in the repository.
