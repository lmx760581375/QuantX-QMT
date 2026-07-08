# QuantX Meta Snapshots

This directory keeps small, canonical metadata snapshots that should travel with the QuantX repo.

## Files

- `snapshots/security_master.csv`
  - Source: `../myquant-strategy/baodata/universe/all_stocks.csv`
  - Snapshot date: `2026-06-25`
  - Coverage: 5188 securities
  - Main columns: `symbol`, `name`, `exchange`, `board`, `baostock_code`, `trade_status`

- `snapshots/industry_membership.csv`
  - Source: `../myquant-strategy/data/stock_industry_map.csv`
  - Snapshot date: `2026-06-25`
  - Coverage: 4039 symbol-industry memberships, 86 industries
  - Main columns: `symbol`, `name`, `industry_source`, `level`, `industry_code`, `industry_name`

- `snapshots/etf_master.csv`
  - Source: AkShare `fund_etf_category_sina`, categories `ETF基金` and `LOF基金`
  - Snapshot date: `2026-07-08`
  - Coverage: 1972 funds, including 1130 industry/theme dynamic-pool candidates
  - Main columns: `symbol`, `name`, `exchange`, `fund_market_type`, `category`, `special_group`, `theme_group`, `is_dynamic_theme_candidate`, latest quote fields, `source`, `snapshot_date`
  - Categories: `sector_theme`, `broad_index`, `cross_border`, `style`, `bond`, `money`, `commodity`

`quantx_meta.sqlite` is generated from these CSV files and is intentionally not the canonical source.

## Import

```bash
conda run -n test python -m quantx.tools.update_meta \
  --source myquant-strategy \
  --security-master-csv data/meta/snapshots/security_master.csv \
  --industry-csv data/meta/snapshots/industry_membership.csv \
  --json
```

The importer uses each CSV row's `snapshot_date` column first. `--date` is only needed when importing a file
without a snapshot date column or when intentionally overriding the imported snapshot date.

The industry snapshot contains 364 rows whose stock names are still blank after backfilling from `security_master.csv`.
Those symbols also are not present in the `all_stocks.csv` source, so QuantX preserves the industry membership instead of inventing names.

## ETF Master Snapshot

Refresh the ETF metadata snapshot with:

```bash
conda run -n test python -m quantx.tools.sync_etf_meta \
  --output data/meta/snapshots/etf_master.csv \
  --json
```

The ETF snapshot is used by ETF rotation experiments to avoid hardcoding industry/theme ETF names. It is not a full point-in-time historical metadata table: fund names and category labels come from the current snapshot. Backtests still require local daily bars before a symbol can be selected or traded.
