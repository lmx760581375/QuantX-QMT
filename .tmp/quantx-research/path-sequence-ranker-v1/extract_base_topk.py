from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    current_session: str | None = None
    current_rows: list[dict[str, Any]] = []
    input_count = 0
    output_count = 0
    sessions = 0
    year_counts: dict[str, int] = {}

    def flush_session() -> None:
        nonlocal writer, output_count, sessions, current_rows
        if not current_rows:
            return
        ranked = sorted(current_rows, key=lambda row: (-row["score"], row["instrument"]))[: args.topk]
        session = ranked[0]["session"]
        year_counts[session[:4]] = year_counts.get(session[:4], 0) + 1
        sessions += 1
        for rank, row in enumerate(ranked, start=1):
            row["base_rank"] = rank
        frame = pd.DataFrame(ranked)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output, table.schema, compression="zstd")
        writer.write_table(table)
        output_count += len(ranked)
        current_rows = []

    with source.open("rb") as handle:
        for record in ijson.items(handle, "records.item"):
            input_count += 1
            signal = record["signal_time"]
            session = str(signal["session"])
            if current_session is None:
                current_session = session
            elif session != current_session:
                flush_session()
                current_session = session
            current_rows.append(
                {
                    "session": session,
                    "instrument": str(record["instrument"]),
                    "score": float(record["score"]),
                    "fold_id": str(record["fold_id"]),
                    "evaluation_tier": str(record.get("evaluation_tier", "development_oos")),
                    "prediction_horizon": int(record["prediction_horizon"]),
                }
            )
    flush_session()
    if writer is not None:
        writer.close()

    checksum = "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        "source": str(source),
        "output": str(output),
        "output_checksum": checksum,
        "topk": int(args.topk),
        "input_records": int(input_count),
        "output_records": int(output_count),
        "sessions": int(sessions),
        "sessions_by_year": dict(sorted(year_counts.items())),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
