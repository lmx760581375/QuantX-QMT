# Historical ST Security State

This directory is a Git-tracked reference dataset built from Tushare Pro's
`stock_st` endpoint by `python -m quantx.tools.sync_st_history`.

Files:

- `st_observed.csv`: rows returned directly by the provider.
- `coverage.csv`: one row per Qlib trading date with fetch and imputation status.
- `st_daily.csv`: effective daily ST membership used by research.
- `st_intervals.csv`: consecutive trading-day intervals for efficient joins.
- `metadata.json`: source, coverage, counts, checksums, and construction notes.

Runtime files such as `sync.log` and `sync.pid` are intentionally excluded
from Git; they are not part of the reference dataset.

The provider sometimes returns an empty response on confirmed trading days.
After the first observed date, those dates inherit the previous observed ST
set. Such rows retain `source_trade_date` and `is_imputed=true`. Dates before
the first observation remain unknown and are never treated as confirmed
non-ST dates.

Credentials are supplied through `TUSHARE_TOKEN` or `TS_TOKEN` and are never
stored here.
