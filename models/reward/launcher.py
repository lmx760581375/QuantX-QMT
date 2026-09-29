"""用版本化 YAML 启动 Reward 训练或推理。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from quantx_reward.config import args_mapping_to_cli, load_yaml, print_json, resolve_repo_path


def build_command(
    config_path: str | Path,
    *,
    nproc_per_node: int | None = None,
    nnodes: int = 1,
    node_rank: int | None = None,
    rdzv_endpoint: str | None = None,
    rdzv_id: str | None = None,
) -> list[str]:
    config = load_yaml(config_path)
    module = str(config.get("module") or "")
    if not module:
        raise ValueError("Model launch config requires module")
    args = dict(config.get("args") or {})
    positional = ("command",) if module.endswith(".train") else ()
    module_args = args_mapping_to_cli(args, positional_keys=positional)
    nproc = int(nproc_per_node or config.get("nproc_per_node", 1))
    nodes = int(nnodes)
    if nproc <= 1 and nodes <= 1:
        return [sys.executable, "-m", module, *module_args]
    distributed = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
    ]
    if nodes <= 1:
        distributed.append("--standalone")
    else:
        if node_rank is None or not rdzv_endpoint:
            raise ValueError("Multi-node launch requires node_rank and rdzv_endpoint")
        distributed.extend([
            f"--nnodes={nodes}",
            f"--node_rank={int(node_rank)}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={rdzv_endpoint}",
            f"--rdzv_id={rdzv_id or 'quantx-reward'}",
        ])
    return [*distributed, "-m", module, "--", *module_args]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--nproc-per-node", type=int)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--rdzv-endpoint")
    parser.add_argument("--rdzv-id")
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args(argv)
    command = build_command(
        resolve_repo_path(parsed.config),
        nproc_per_node=parsed.nproc_per_node,
        nnodes=parsed.nnodes,
        node_rank=parsed.node_rank,
        rdzv_endpoint=parsed.rdzv_endpoint,
        rdzv_id=parsed.rdzv_id,
    )
    if parsed.dry_run:
        print_json({"ok": True, "command": command})
        return 0
    return int(subprocess.run(command, cwd=resolve_repo_path("."), check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
