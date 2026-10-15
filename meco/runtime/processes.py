"""Lifecycle helpers for this coordinator's own local services."""
import asyncio
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def assert_ports_free(ports):
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
    finally:
        for sock in sockets:
            sock.close()


async def ready(url, process, client, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited: {process.returncode}")
        try:
            response = await client.get(url + "/health", timeout=2)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        await asyncio.sleep(.5)
    raise TimeoutError("model service did not become ready")


async def connect(port, process, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"MCU service exited: {process.returncode}")
        try:
            return await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await asyncio.sleep(.2)
    raise TimeoutError("MCU UART did not become ready")


class OwnedProcesses:
    def __init__(self, output):
        self.output = Path(output)
        self.processes = []
        self.logs = []

    def launch(self, name, command, affinity, env, cwd):
        log = (self.output / (name + ".log")).open("w", encoding="utf-8")
        self.logs.append(log)
        process = subprocess.Popen(
            ["taskset", "-c", ",".join(map(str, affinity)), *command],
            cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        self.processes.append(process)
        return process

    async def close(self):
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
                try:
                    await asyncio.to_thread(process.wait, timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    await asyncio.to_thread(process.wait)
        for log in self.logs:
            log.close()
