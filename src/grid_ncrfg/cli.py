"""Command-line interface for pl-grid-connection-ncrfg."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .powerflow import verify_two_bus_analytical
from .pipeline import run_full_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grid-ncrfg",
        description="Polish MV/110 kV grid connection study, NC RfG Q(U)/cos-phi(P) reactive control, and non-firm hosting capacity.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run the full study, generate figures, results, and dashboard.")
    p_run.add_argument("--out", type=Path, default=None, help="Output repository root directory.")

    p_ver = sub.add_parser("verify", help="Verify Newton-Raphson Ybus solver against exact 2-bus analytical solution.")
    p_ver.add_argument("--out", type=Path, default=None, help="Optional JSON path to write verification summary.")

    p_site = sub.add_parser("site", help="Build figures in docs/img/ and interactive dashboard in docs/.")
    p_site.add_argument("--out", type=Path, default=None, help="Output repository root directory.")

    args = parser.parse_args(argv)
    cmd = args.cmd or "run"

    if cmd == "verify":
        ver = verify_two_bus_analytical()
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(ver, indent=2), encoding="utf-8")
        print(json.dumps(ver, indent=2))
        return 0

    summary = run_full_pipeline(output_root=args.out)
    print(
        f"Completed grid connection study: F1 hosting capacity "
        f"{summary['headline']['f1_uncontrolled_mw']} MW (uncontrolled) -> "
        f"{summary['headline']['f1_cosphi_mw']} MW (cos-phi=0.95) -> "
        f"{summary['headline']['f1_qu_mw']} MW (NC RfG Q(U)) -> "
        f"{summary['headline']['f1_nonfirm70_mw']} MW (70% cap)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
