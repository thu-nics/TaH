#!/usr/bin/env python
"""Launch one or more mini-sglang API servers for online evaluation.

Usage:
    python script2/eval/launch_minisgl_server.py \
    --model_path output/checkpoint \
    --ports 30010 30011 30012 30013 30014 30015 30016 30017 \
    --base_gpu 0 \
    --distributed_port_base 31010 \
    --tah_iter_threshold 0.5 \
    --tah_iter_decision sample \
    --kill_existing \
    --wait
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


def _pids_on_port(port: int) -> list:
    # lsof is not installed on every eval machine; fall back to ss.
    for cmd, parse in (
        (["lsof", "-t", "-iTCP:" + str(port), "-sTCP:LISTEN"], lambda out: out.split()),
        (
            ["ss", "-tlnp", f"sport = :{port}"],
            lambda out: re.findall(r"pid=(\d+)", out),
        ),
    ):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            continue
        pids = [int(p) for p in parse(result.stdout)]
        if pids or result.returncode == 0:
            return pids
    print(f"[launch] neither lsof nor ss available; cannot scan port {port}")
    return []


def kill_process_on_port(port: int) -> None:
    pids = _pids_on_port(port)
    if not pids:
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + 10
    while time.time() < deadline:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            return
        time.sleep(0.2)

    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def wait_for_server(base_url: str, timeout: int = 600, process=None) -> None:
    deadline = time.time() + timeout
    models_url = f"{base_url.rstrip('/')}/v1/models"
    opener = build_opener(ProxyHandler({}))
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Server at {base_url} exited with code {process.returncode}")
        try:
            with opener.open(models_url, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return
        except URLError:
            time.sleep(2)
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Server at {base_url} not ready within {timeout}s")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch mini-sglang servers for TaH2 online evaluation.")
    parser.add_argument("--model_path", type=str, required=True, help="Model/checkpoint path to serve.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host.")
    parser.add_argument("--ports", type=int, nargs="+", required=True, help="One port per server.")
    parser.add_argument(
        "--gpu_groups",
        type=str,
        nargs="+",
        default=None,
        help='CUDA_VISIBLE_DEVICES spec per server, e.g. "0" "1" or "0,1" "2,3".',
    )
    parser.add_argument("--base_gpu", type=int, default=0, help="Base GPU index if --gpu_groups is not provided.")
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help=(
            "Tensor-parallel size inside each server. Inferred from --gpu_groups "
            "when provided; otherwise defaults to 1."
        ),
    )
    parser.add_argument(
        "--distributed_port_base",
        type=int,
        required=True,
        help=(
            "Base TCP port for intra-server distributed init. Required; choose "
            "ports that do not overlap with API server ports."
        ),
    )
    parser.add_argument("--dtype", type=str, default=None, help="Optional dtype passed to minisgl.")
    parser.add_argument("--max_seq_len_override", type=int, default=None)
    parser.add_argument("--cuda_graph_max_bs", type=int, default=None)
    parser.add_argument("--page_size", type=int, default=None)
    parser.add_argument(
        "--memory_ratio",
        type=float,
        default=None,
        help="Fraction of GPU memory used for the KV cache (forwarded as --memory-ratio).",
    )
    parser.add_argument(
        "--cache_type",
        default=None,
        help="Prefix cache type forwarded as --cache-type ('radix' default; "
             "'naive' disables prefix reuse — for cache-sensitivity A/Bs).",
    )
    parser.add_argument("--tah_max_iter", type=int, default=None, help="TAH max_iter override for mini-sglang.")
    parser.add_argument("--tah_iter_threshold", type=float, default=None, help="TAH threshold at server startup.")
    parser.add_argument("--tah_duo_iter1_reserve_factor", type=float, default=None, help="TAH duo iter1 reserve factor.")
    parser.add_argument(
        "--tah_dynamic_preempt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forward --tah-dynamic-preempt to the server (default ON). DUO "
            "TaH only: greedy admission + runtime preemption replaces the static "
            "reserve, so --tah_duo_iter1_reserve_factor is then ignored. Pass "
            "--no-tah_dynamic_preempt to fall back to the static reserve path. "
            "No-op for non-duo checkpoints."
        ),
    )
    parser.add_argument("--tah_iter_decision", choices=["threshold", "sample"], default=None)
    parser.add_argument("--kill_existing", action="store_true", help="Kill processes already listening on target ports.")
    parser.add_argument("--wait", action="store_true", help="Wait until all servers respond on /v1/models.")
    parser.add_argument("--wait_timeout", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    ports = list(args.ports)
    if not ports:
        raise ValueError("At least one port is required.")
    if len(set(ports)) != len(ports):
        raise ValueError("--ports must contain distinct ports.")
    if args.tp_size is not None and args.tp_size < 1:
        raise ValueError("--tp_size must be >= 1.")

    if args.gpu_groups is not None:
        gpu_groups = list(args.gpu_groups)
        if len(gpu_groups) != len(ports):
            raise ValueError("--gpu_groups must have the same length as --ports.")
    else:
        tp_size = args.tp_size or 1
        gpu_groups = [
            ",".join(str(args.base_gpu + i * tp_size + j) for j in range(tp_size))
            for i in range(len(ports))
        ]

    tp_sizes = []
    for group in gpu_groups:
        devices = group.split(",")
        if any(not device.strip() for device in devices) or len(set(devices)) != len(devices):
            raise ValueError("Each --gpu_groups entry must list distinct GPU devices.")
        if args.tp_size is not None and len(devices) != args.tp_size:
            raise ValueError("--tp_size must match the number of GPUs in each --gpu_groups entry.")
        tp_sizes.append(len(devices))

    local_model = Path(args.model_path).expanduser()
    model_path = str(local_model.resolve()) if local_model.exists() else args.model_path
    dist_ports = [args.distributed_port_base + i for i in range(len(ports))]
    if any(not 1 <= port <= 65535 for port in ports + dist_ports):
        raise ValueError("API and distributed ports must be in [1, 65535].")
    overlap = sorted(set(ports) & set(dist_ports))
    if overlap:
        raise ValueError(
            "API ports and distributed ports must not overlap. "
            f"Overlapping port(s): {overlap}"
        )

    processes = []
    base_urls = [f"http://{args.host}:{port}" for port in ports]

    print(f"Launching {len(ports)} mini-sglang servers")
    print(f"Model: {model_path}")
    if args.tah_dynamic_preempt and args.tah_duo_iter1_reserve_factor is not None:
        print(
            "[warn] --tah-dynamic-preempt is ON; the server forces the static reserve "
            f"to 0, so --tah_duo_iter1_reserve_factor {args.tah_duo_iter1_reserve_factor} "
            "is ignored. Pass --no-tah_dynamic_preempt to use the static reserve path.",
            file=sys.stderr,
        )

    try:
        for idx, (port, gpu_group, dist_port) in enumerate(zip(ports, gpu_groups, dist_ports)):
            if args.kill_existing:
                kill_process_on_port(port)

            cmd = [
                sys.executable,
                "-m",
                "tah2.minisgl",
                "--model-path",
                model_path,
                "--host",
                args.host,
                "--port",
                str(port),
                "--distributed-port",
                str(dist_port),
                "--dp",
                str(idx),
            ]
            cmd.extend(["--tp-size", str(tp_sizes[idx])])
            if args.dtype is not None:
                cmd.extend(["--dtype", args.dtype])
            if args.max_seq_len_override is not None:
                cmd.extend(["--max-seq-len-override", str(args.max_seq_len_override)])
            if args.cuda_graph_max_bs is not None:
                cmd.extend(["--cuda-graph-max-bs", str(args.cuda_graph_max_bs)])
            if args.page_size is not None:
                cmd.extend(["--page-size", str(args.page_size)])
            if args.memory_ratio is not None:
                cmd.extend(["--memory-ratio", str(args.memory_ratio)])
            if args.cache_type is not None:
                cmd.extend(["--cache-type", args.cache_type])
            if args.tah_max_iter is not None:
                cmd.extend(["--tah-max-iter", str(args.tah_max_iter)])
            if args.tah_iter_threshold is not None:
                cmd.extend(["--tah-iter-threshold", str(args.tah_iter_threshold)])
            if args.tah_iter_decision is not None:
                cmd.extend(["--tah-iter-decision", args.tah_iter_decision])
            if args.tah_duo_iter1_reserve_factor is not None:
                cmd.extend(["--tah-duo-iter1-reserve-factor", str(args.tah_duo_iter1_reserve_factor)])
            if args.tah_dynamic_preempt:
                cmd.append("--tah-dynamic-preempt")

            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu_group

            print(f"  server[{idx}] url={base_urls[idx]} gpus={gpu_group} dist_port={dist_port}")
            proc = subprocess.Popen(cmd, env=env, start_new_session=True)
            processes.append(proc)

        if args.wait:
            for base_url, proc in zip(base_urls, processes):
                print(f"Waiting for {base_url} ...")
                wait_for_server(base_url, timeout=args.wait_timeout, process=proc)

        print("Ready servers:")
        for base_url in base_urls:
            print(f"  {base_url}")
        print("Press Ctrl+C to stop.")

        for proc in processes:
            returncode = proc.wait()
            if returncode:
                raise RuntimeError(f"Server exited with code {returncode}")
    except KeyboardInterrupt:
        print("\nStopping servers...")
    finally:
        for proc in processes:
            # The scheduler and tokenizer are children of the API process.
            # Terminate the whole session even if the API process already exited.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc in processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


if __name__ == "__main__":
    main()
