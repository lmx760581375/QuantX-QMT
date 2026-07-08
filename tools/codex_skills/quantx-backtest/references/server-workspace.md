# Web Workspace

Use this reference when starting, restarting, or checking the QuantX local web workspace.

## Check Current Server

```bash
ps -ef | rg 'quantx\.server\.app' | rg -v 'rg'
ss -ltnp 'sport = :8000' || true
```

## Restart Server

```bash
bash tools/codex_skills/quantx-backtest/scripts/restart_server.sh /home/users/mingxiao.li/git/quantization/quantx
```

Manual fallback:

```bash
kill <pid>
setsid /home/users/mingxiao.li/anaconda3/envs/test/bin/python -m quantx.server.app \
  > /tmp/quantx_server.log 2>&1 < /dev/null &
tail -n 40 /tmp/quantx_server.log
```

## URL

```text
http://127.0.0.1:8000
```

## API Smoke

```bash
curl -s http://127.0.0.1:8000/api/meta/symbols/SH600000
curl -s http://127.0.0.1:8000/api/reports
```

## Symbol Page

The per-symbol K-line page should show stock name and industry. The backend endpoint is:

```text
GET /api/reports/{run_id}/symbols/{symbol}
```

Expected useful fields: `name`, `meta.industry_name`, `bars`, `trades`, `round_trips`.
