from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantx.core.data.qmt_client import QMTClient  # noqa: E402


def normalize(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main() -> None:
    stocks = ["600000.SH", "000001.SZ", "002594.SZ"]
    with QMTClient(pause_seconds=0, max_retries=1) as client:
        xtdata = client.xtdata
        full_tick = xtdata.get_full_tick(stocks)
        print("FULL_TICK_DETAIL")
        for stock, tick in full_tick.items():
            print("\n" + stock)
            for key in [
                "timetag",
                "lastPrice",
                "lastClose",
                "open",
                "high",
                "low",
                "amount",
                "volume",
                "stockStatus",
                "askPrice",
                "bidPrice",
                "askVol",
                "bidVol",
            ]:
                value = normalize(tick.get(key))
                print(f"{key}: {type(value).__name__}: {value}")

        print("\nALT_CALLS")
        for name in ["get_fullspeed_orderbook", "get_transactioncount"]:
            if not hasattr(xtdata, name):
                print(f"{name}: MISSING")
                continue
            fn = getattr(xtdata, name)
            for arg in ["600000.SH", ("600000.SH",), ["600000.SH"]]:
                try:
                    value = fn(arg)
                    print(f"{name} {arg!r}: OK {type(value).__name__} {json.dumps(value, ensure_ascii=False, default=str)[:1200]}")
                except Exception as exc:
                    print(f"{name} {arg!r}: FAIL {type(exc).__name__} {str(exc).splitlines()[0][:500]}")


if __name__ == "__main__":
    main()
