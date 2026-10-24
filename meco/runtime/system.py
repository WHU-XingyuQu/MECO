"""Launch the runtime modules with explicit paths and independent process ownership."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import re
import subprocess

import httpx

from meco.device.uart import MaintenanceMCULink, observation_fields
from meco.strategy.routing import requires_gpu
from .pipeline import CascadePipeline
from .processes import OwnedProcesses, assert_ports_free, connect, ready

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value, *, preserve_symlinks=False):
    value = os.path.expandvars(os.path.expanduser(value))
    if "$" in value:
        raise ValueError("unresolved environment variable in runtime path")
    path = Path(value).absolute()
    resolved = path.resolve(strict=True)
    for candidate in (path, resolved):
        if any(character in str(candidate) for character in ('\n', '\r', '"', '@')) or re.search(r"\s", str(candidate)):
            raise ValueError("runtime/Renode paths must not contain whitespace or monitor metacharacters")
    # Python locates pyvenv.cfg relative to the invoked executable. Resolving a
    # venv's interpreter symlink would silently select the base environment.
    return path if preserve_symlinks else resolved


def validate_tasks(tasks):
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must contain a nonempty requests list")
    uids, identities = set(), set()
    for task in tasks:
        uid = task["uid"]
        if not isinstance(uid, str) or not uid or uid in uids:
            raise ValueError("duplicate or invalid task uid")
        uids.add(uid)
        device = task["device_id"]
        if type(device) is not int or device not in (1, 2, 3):
            raise ValueError("device_id must be 1, 2 or 3")
        if task["family"] != {1: "conveyor", 2: "pump", 3: "robot"}[device]:
            raise ValueError("task family and device disagree")
        sequence = task["sample_sequence"]
        identity = device, sequence
        if type(sequence) is not int or not 0 < sequence <= 0xFFFFFFFF or identity in identities:
            raise ValueError("duplicate or invalid MCU sample identity")
        identities.add(identity)
        arrival = task.get("arrival_ms", 0)
        if type(arrival) not in (int, float) or not math.isfinite(arrival) or arrival < 0:
            raise ValueError("arrival_ms must be finite and nonnegative")
        ids = task["input_ids"]
        if not isinstance(ids, list) or not 0 < len(ids) <= 2048 or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError("invalid serialized input_ids")
        if task.get("max_new_tokens", 96) != 96:
            raise ValueError("this runtime uses a 96-token generation budget")
        requires_gpu(task["observation"])
        observation_fields(task)


async def run(config_path, tasks_path, output):
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("the system launcher requires Linux/WSL2")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task_file = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks = task_file["requests"] if isinstance(task_file, dict) else task_file
    validate_tasks(tasks)
    cpu_python, gpu_python = [resolve_path(config[key], preserve_symlinks=True)
                             for key in ("cpu_python", "gpu_python")]
    renode, mapper = [resolve_path(config[key]) for key in ("renode", "mapper_dir")]
    output = output.resolve()
    if re.search(r"\s", str(output)) or any(c in str(output) for c in ('"', '@', '\n', '\r')):
        raise ValueError("use an output path without whitespace or Renode metacharacters")
    resolve_path(str(ROOT))
    port = config.get("port_base", 39800)
    if type(port) is not int or not 1024 <= port <= 65512:
        raise ValueError("invalid base port")
    assert_ports_free([port+i for i in (0, 1, 10, 21, 22, 23)])
    affinity = config.get("affinity", {"cpu": [0,1,2,3], "gpu": [4,5,6,7], "mcu": [8,9]})
    allowed = os.sched_getaffinity(0)
    for name in ("cpu", "gpu", "mcu"):
        if not affinity.get(name) or not set(affinity[name]) <= allowed:
            raise ValueError("invalid process affinity for " + name)
    timeout = config.get("request_timeout_s", 7200)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("invalid request timeout")
    output.mkdir(parents=True, exist_ok=False)
    for device in (1, 2, 3):
        subprocess.run(["bash", str(ROOT / "mcu/build.sh"), str(output / f"firmware-{device}"), str(device)], check=True)
    lines = []
    for device in (1, 2, 3):
        lines.extend([f'mach create "maintenance-{device}"',
                      f'machine LoadPlatformDescription @{ROOT / "mcu/renode/board.repl"}',
                      f'sysbus LoadELF @{output / f"firmware-{device}/device.elf"}',
                      f'emulation CreateServerSocketTerminal {port+20+device} "uart-{device}" false',
                      f'connector Connect sysbus.usart2 "uart-{device}"'])
    scenario = output / "system.resc"
    scenario.write_text("\n".join(lines+["start"]) + "\n", encoding="utf-8")
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    owner, links = OwnedProcesses(output), {}
    try:
        cpu = owner.launch("cpu", [str(cpu_python), "-m", "meco.inference.service", "--tier", "cpu", "--port", str(port)],
                           affinity["cpu"], {**environment, "CUDA_VISIBLE_DEVICES": ""}, ROOT)
        gpu = owner.launch("gpu", [str(gpu_python), "-m", "meco.inference.service", "--tier", "gpu", "--port", str(port+1),
                                   "--mapper", str(mapper)], affinity["gpu"], environment, ROOT)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            cpu_url, gpu_url = f"http://127.0.0.1:{port}", f"http://127.0.0.1:{port+1}"
            await asyncio.gather(ready(cpu_url, cpu, client), ready(gpu_url, gpu, client))
            mcu = owner.launch("mcu", [str(renode), "--disable-gui", "--plain", "--port", str(port+10),
                                      "--config", str(output / "renode.config"), str(scenario)],
                               affinity["mcu"], environment, renode.parent)
            for device in (1, 2, 3):
                reader, writer = await connect(port+20+device, mcu)
                link = MaintenanceMCULink(reader, writer, device=device, session=device)
                links[device] = link
                await link.start()
            pipeline = CascadePipeline(client, cpu_url, gpu_url, links)
            with (output / "responses.jsonl").open("x", encoding="utf-8") as stream:
                def emit(result):
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(json.dumps({"uid": result["uid"], "status": result["status"]}), flush=True)
                return await pipeline.run_tasks(tasks, emit)
    finally:
        for link in links.values():
            try:
                await link.aclose()
            except OSError:
                pass
        await owner.close()


def main():
    parser = argparse.ArgumentParser(description="Run the complete-KV MCU/CPU/GPU maintenance pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = asyncio.run(run(args.config, args.tasks, args.output))
    if any(row["status"] in ("error", "timeout") for row in results):
        raise SystemExit(1)
