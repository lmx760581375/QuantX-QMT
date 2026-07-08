# Safety Rules

Use this reference before destructive, data-changing, or long-running QuantX work.

## Do Not Do These Unless Explicitly Requested

- Do not delete `runs/`, `data/`, `data/qlib_data*`, or user configs.
- Do not reset git state or revert unrelated files.
- Do not overwrite an existing strategy config without explaining the change.
- Do not treat `data/meta/quantx_meta.sqlite` as the only trusted metadata source.
- Do not default to online metadata APIs.
- Do not promise that future industry/group factor YAML is runnable until implementation is verified.

## Required Habits

- Generate experimental configs under `configs/strategies/generated/`.
- Dry-run a generated or changed config before full backtest.
- State when a full-market multi-year backtest may take a while.
- Prefer JSON outputs over parsing free-form logs.
- Include exact commands and key numeric outputs in final summaries.
- For result mismatch, follow `diagnostics.md` before claiming a bug.

## Local Environment

- Use `conda run -n test python`.
- Use company pip mirror if installing packages: `https://pypi.hobot.cc/simple`.
- Network is slow and online data APIs are unreliable.
