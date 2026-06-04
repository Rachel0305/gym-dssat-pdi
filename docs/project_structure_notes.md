# Project Structure Notes

Date: 2026-06-04

## Directory Policy

Markdown prompts for Codex tasks are stored in:

```text
prompts/
```

Prompt files should use sequential numbering:

```text
001_topic_action.md
002_topic_action.md
003_topic_action.md
003_01_subtask.md
```

Project notes, analysis records, experiment designs, troubleshooting records, and duplicated report copies are stored in:

```text
docs/
```

Documentation files should use lowercase English names with underscores. Avoid names such as `final`, `v2`, or `final2`.

## Current Documentation Archive

PPT reports copied from their original experiment folders are archived under:

```text
docs/ppt_reports/
```

Markdown experiment records copied from output folders are archived under:

```text
docs/experiment_records/
```

The original files were not deleted or moved. These are additional documentation copies for easier Git-based tracking.

## Weather Data

Cleaned daily weather data are stored in:

```text
weather_clean/
```

Daily weather CSV files now include:

```text
year_doy
```

`year_doy` is a compact date key in `YYYYDDD` format, where `YYYY` is the calendar year and `DDD` is day of year with three digits. Example:

```text
2007001
```

This field is useful for DSSAT-style date joins, WTH generation, and station-year diagnostics.

## Git Notes

The latest organized documentation and initialization updates were pushed to:

```text
codex/soil-water-initialization-backup
```

This branch was used to avoid overwriting remote changes already present on `codex-reward-sweep-backup`.
