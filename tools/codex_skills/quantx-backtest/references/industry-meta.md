# Industry And Metadata

Use this reference for stock names, industry display, industry filters, and future industry factor configs.

## Current Metadata Source

QuantX currently uses committed CSV snapshots as the reliable metadata source:

- `data/meta/snapshots/security_master.csv`: 5188 securities with stock names.
- `data/meta/snapshots/industry_membership.csv`: 4039 symbol-industry memberships across 86 industries.
- `data/meta/quantx_meta.sqlite`: generated cache; do not treat it as canonical.

Online AkShare/BaoStock metadata endpoints have been unreliable on this machine. Prefer the snapshots.

## Import Command

```bash
conda run -n test python -m quantx.tools.update_meta \
  --source myquant-strategy \
  --security-master-csv data/meta/snapshots/security_master.csv \
  --industry-csv data/meta/snapshots/industry_membership.csv \
  --json
```

The importer preserves CSV row `snapshot_date` when present.

## Query Metadata

```bash
python tools/codex_skills/quantx-backtest/scripts/check_meta.py --root . --symbols SH600000 SH600137 SZ920010
```

Known behavior: some symbols exist in industry membership but not security master. They should still return industry with `name: null`.

## Industry Config Direction

The target config language for industry-aware strategies is:

```yaml
meta:
  industry:
    source: eastmoney
    mode: latest_static

groups:
  industry:
    by: meta.industry_name

group_factors:
  industry_momentum20:
    group: industry
    expr: GroupMean(close / Ref(close, 20) - 1, group=industry)

selector:
  pipeline:
    - where: close > Mean(close, 20)
    - group_rank:
        group: industry
        factor: industry_momentum20
        top: 5
    - rank:
        factor: close / Ref(close, 20) - 1
        top: 10
```

This group-factor schema is a target design. Verify implementation before claiming a config using it is runnable.
