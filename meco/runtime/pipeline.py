"""CPU diagnosis -> optional complete KV handoff -> MCU acknowledgement."""
import asyncio
import time

from meco.device.response import parse_response
from meco.strategy.routing import requires_gpu
from meco.strategy.scheduler import StageScheduler
from meco.transport.frames import digest, pack, unpack


class CascadePipeline:
    def __init__(self, client, cpu_url, gpu_url, links):
        self.client = client
        self.cpu_url, self.gpu_url = cpu_url, gpu_url
        self.links = links
        self.scheduler = StageScheduler()

    async def execute(self, task):
        started = time.perf_counter()
        downstream = requires_gpu(task["observation"])
        record = {"uid": task["uid"], "requires_gpu": downstream, "status": "error"}
        link = self.links[task["device_id"]]
        try:
            record["observed"] = await link.inject(task)
            async with self.scheduler.cpu:
                body = pack({"uid": task["uid"], "input_ids": task["input_ids"],
                             "max_new_tokens": task.get("max_new_tokens", 96),
                             "observation": task["observation"], "transfer": downstream})
                response = await self.client.post(self.cpu_url + "/generate", content=body)
                response.raise_for_status()
                cpu, payload = unpack(response.content)
            if cpu["uid"] != task["uid"] or len(payload) != cpu["kv_bytes"] or digest(payload) != cpu["kv_sha256"]:
                raise ValueError("CPU result identity or KV payload mismatch")
            record["cpu"] = cpu
            result = cpu
            if downstream:
                async with self.scheduler.gpu:
                    response = await self.client.post(self.gpu_url + "/generate", content=pack(cpu, payload))
                    response.raise_for_status()
                    result, unexpected = unpack(response.content)
                if unexpected or result["uid"] != task["uid"] or result["prefix_sha256"] != cpu["prefix_sha256"]:
                    raise ValueError("GPU result identity or prefix mismatch")
                if result["upstream_tokens_replayed"] != 0 or result["cached_tokens"] != len(cpu["prefix_ids"]):
                    raise ValueError("complete KV continuation contract violated")
                record["gpu"] = result
            record["text"] = result["text"]
            try:
                decoded, _ = parse_response(result["text"])
            except (ValueError, TypeError) as error:
                decoded = {"diagnosis": "INVALID_RESPONSE", "action": "INVALID_RESPONSE"}
                record["parse_error"] = str(error)
            record["response"] = decoded
            ack = await link.complete(task, decoded["diagnosis"], decoded["action"])
            record["acknowledgement"] = ack
            record["status"] = "completed" if ack[7] == "OK" else "rejected"
        except Exception as error:
            record.update(status="timeout" if isinstance(error, TimeoutError) else "error",
                          error_type=type(error).__name__, error=str(error))
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return record

    async def run_tasks(self, tasks, emit):
        origin = time.perf_counter()

        async def one(task):
            await asyncio.sleep(max(0, origin + task.get("arrival_ms", 0) / 1000 - time.perf_counter()))
            result = await self.execute(task)
            emit(result)
            return result

        return await asyncio.gather(*(one(task) for task in tasks))
