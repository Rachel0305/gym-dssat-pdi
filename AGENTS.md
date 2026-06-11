# AGENTS.md

## Safety rules: file and system protection

This project contains important research data. Be extremely conservative with file operations.

### Forbidden actions

Never run or suggest destructive system-level commands, including but not limited to:

- Formatting disks or partitions
- Deleting files outside the current project directory
- Deleting the user's home directory, Desktop, Downloads, Documents, external drives, or any parent directory of the project
- Running commands like `rm -rf /`, `rm -rf ~`, `rm -rf ..`, `del /s`, `rmdir /s`, `format`, `diskpart`, `mkfs`, `dd`, or PowerShell destructive commands
- Changing system environment variables permanently
- Installing, uninstalling, or upgrading system-wide software without explicit user approval
- Modifying Docker, WSL, Git, SSH, or system configuration outside this project unless explicitly requested
- Pushing to GitHub without user confirmation
- Force-pushing, rebasing public branches, or deleting Git branches without user confirmation

### Directory boundary

Only read and modify files inside the current project directory.

Do not access, modify, move, rename, or delete files outside the project directory.

If a required file appears to be outside the project directory, stop and ask the user to copy it into the project directory.

### Data protection

The following files and folders are important research data and must not be deleted or overwritten:

- `my_data/`
- original `.xls` / `.xlsx` files
- original `.WTH` files
- original `.SOL` files
- original `.jinja2` files
- `MZCER048.CUL`
- existing training scripts
- existing reward functions
- existing wrapper files
- existing evaluation results

Before modifying any important file, create a backup copy under:

```text
backups/
```

## Project context

This project uses gym-DSSAT / DSSAT for maize water-nitrogen management optimization across five Chinese ecological stations.

## General rules

- Communicate with the user in Chinese.
- Do not overwrite existing working scripts without backup.
- Keep station-specific training, evaluation, and plotting scripts separate when possible.
- Save all intermediate CSV, logs, figures, and experiment records.
- Record failed attempts, errors, fixes, and experiment decisions.
- Do not upload large model files, cache files, or temporary files to GitHub.
- Before long training, first run a small smoke test.