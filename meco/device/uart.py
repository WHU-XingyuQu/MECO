"""Bounded protocol-v3 tickets for concurrent simulated maintenance observations."""
from __future__ import annotations

import asyncio
import re
import time

from .crc import crc16, integer


DIAGNOSES = ("NORMAL", "SENSOR_FAULT", "MOTOR_OVERLOAD", "BELT_JAM",
             "BEARING_OVERHEAT", "CAVITATION", "THERMAL_OVERLOAD", "DRIVE_FAULT")
ACTIONS = ("KEEP", "CHECK_SENSOR", "INSPECT_MOTOR", "INSPECT_BELT",
           "INSPECT_BEARING", "CHECK_INLET", "COOL_JOINT", "INSPECT_DRIVE")
MAX_FRAME = 160
COUNTS = {"H": 4, "R": 7, "Q": 10, "T": 11, "D": 7, "A": 11, "E": 5}


def fields(payload):
    row = payload.split(",")
    if not row or row[0] not in COUNTS or len(row) != COUNTS[row[0]] or row[1] != "3":
        raise ValueError("invalid maintenance frame schema")
    integer(row[2], 1, 3)
    integer(row[3], 1)
    if row[0] in {"Q", "T", "D", "A"}:
        integer(row[4], 1)
    if row[0] in {"Q", "T"}:
        context = integer(row[5], 1, 12)
        if (context - 1) // 4 + 1 != int(row[2]):
            raise ValueError("context belongs to another MCU")
        integer(row[6], -400, 1500)
        integer(row[7], 0, 65535)
        integer(row[8], 0, 65535)
        integer(row[9], 0, 1)
    if row[0] == "T":
        integer(row[10])
    if row[0] in {"D", "A"}:
        integer(row[5], 0, 255)
        integer(row[6], 0, 255)
    if row[0] == "A":
        if row[7] not in {"OK", "REJECTED"}:
            raise ValueError("unknown maintenance acknowledgement")
        for value in row[8:]:
            integer(value)
    if row[0] == "R":
        if row[4] != "READY":
            raise ValueError("invalid maintenance handshake")
        integer(row[5])
        integer(row[6])
    if row[0] == "E" and row[4] not in {"FRAME", "CRC", "SESSION", "RANGE", "DUPLICATE",
                                          "CAPACITY", "TASK", "OVERFLOW"}:
        raise ValueError("invalid maintenance error")
    return tuple(row)


def encode(*values):
    payload = ",".join(map(str, values))
    fields(payload)
    data = payload.encode("ascii")
    frame = data + f"*{crc16(data):04X}\n".encode()
    if len(frame) > MAX_FRAME:
        raise ValueError("maintenance frame exceeds MCU buffer")
    return frame


def decode(frame):
    if type(frame) is not bytes or len(frame) > MAX_FRAME or not frame.endswith(b"\n"):
        raise ValueError("invalid maintenance frame size")
    payload, separator, checksum = frame[:-1].rpartition(b"*")
    if not separator or not re.fullmatch(rb"[0-9A-F]{4}", checksum) or crc16(payload) != int(checksum, 16):
        raise ValueError("maintenance CRC mismatch")
    return fields(payload.decode("ascii"))


def observation_fields(row):
    obs, family = row["observation"], row["family"]
    if family == "conveyor":
        temperature, signal, auxiliary = obs["motor_temp_c"], obs["current_a"], obs["belt_speed_mps"]
    elif family == "pump":
        temperature, signal, auxiliary = obs["bearing_temp_c"], obs["pressure_bar"], obs["vibration_mms"]
    else:
        temperature, signal, auxiliary = obs["joint_temp_c"], obs["torque_percent"], obs["tracking_error_deg"]
    variant = int(row["document_id"].rsplit("-", 1)[1]) - 1
    return ((row["device_id"] - 1) * 4 + variant + 1, round(temperature * 10),
            round(signal * 100), round(auxiliary * 100), int(obs["sensor_valid"]))


class MaintenanceMCULink:
    def __init__(self, reader, writer, *, device, session, emit=None):
        self.reader, self.writer, self.device, self.session = reader, writer, device, session
        self.emit = emit or (lambda *args: None)
        self.observations, self.completions = {}, {}
        self.lock = asyncio.Lock()
        self.ready = asyncio.get_running_loop().create_future()
        self.reader_task = None

    async def send(self, frame):
        async with self.lock:
            self.writer.write(frame)
            await self.writer.drain()
            self.emit("uart_tx", {"device": self.device, "frame": frame.decode().strip(),
                                  "time_ms": time.perf_counter() * 1000})

    async def start(self):
        self.reader_task = asyncio.create_task(self._read())
        deadline = time.monotonic() + 10
        # Renode binds the socket before guest firmware enables USART reception.
        # Retrying the idempotent handshake avoids treating socket-open as boot-ready.
        while time.monotonic() < deadline:
            await self.send(encode("H", 3, self.device, self.session))
            try:
                return await asyncio.wait_for(asyncio.shield(self.ready), .25)
            except TimeoutError:
                pass
        raise TimeoutError("MCU firmware handshake did not complete")

    async def _read(self):
        try:
            while frame := await self.reader.readline():
                row = decode(frame)
                if (int(row[2]), int(row[3])) != (self.device, self.session):
                    raise ValueError("cross-device/session maintenance frame")
                self.emit("uart_rx", {"device": self.device, "frame": frame.decode().strip(),
                                      "time_ms": time.perf_counter() * 1000})
                if row[0] == "R" and not self.ready.done():
                    self.ready.set_result(row)
                elif row[0] in {"T", "A"}:
                    mapping = self.observations if row[0] == "T" else self.completions
                    future = mapping.pop(int(row[4]), None)
                    if future is None or future.done():
                        raise ValueError("unmatched maintenance transaction")
                    future.set_result(row)
                elif row[0] == "E":
                    raise RuntimeError("MCU rejected frame: " + row[4])
        except Exception as exc:
            for future in (self.ready, *self.observations.values(), *self.completions.values()):
                if not future.done():
                    future.set_exception(exc)

    async def inject(self, task):
        if len(self.observations) >= 128:
            raise RuntimeError("MCU host observation bound exceeded")
        sequence = task["sample_sequence"]
        expected = observation_fields(task)
        future = asyncio.get_running_loop().create_future()
        self.observations[sequence] = future
        await self.send(encode("Q", 3, self.device, self.session, sequence, *expected))
        observed = await asyncio.wait_for(future, 10)
        if tuple(map(int, observed[5:10])) != expected:
            raise ValueError("MCU observed telemetry differs from task input")
        return observed

    async def complete(self, task, diagnosis, action):
        sequence = task["sample_sequence"]
        diagnosis_code = DIAGNOSES.index(diagnosis) if diagnosis in DIAGNOSES else 255
        action_code = ACTIONS.index(action) if action in ACTIONS else 255
        future = asyncio.get_running_loop().create_future()
        self.completions[sequence] = future
        await self.send(encode("D", 3, self.device, self.session, sequence, diagnosis_code, action_code))
        acknowledged = await asyncio.wait_for(future, 10)
        if tuple(map(int, acknowledged[5:7])) != (diagnosis_code, action_code):
            raise ValueError("MCU acknowledged different diagnosis/action")
        return acknowledged

    async def aclose(self):
        if self.reader_task:
            self.reader_task.cancel()
            await asyncio.gather(self.reader_task, return_exceptions=True)
        self.writer.close()
        await self.writer.wait_closed()
