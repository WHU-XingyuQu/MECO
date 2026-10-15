"""One FIFO lock per stage; CPU and GPU stages may overlap."""
import asyncio

class StageScheduler:
    def __init__(self):
        self.cpu = asyncio.Lock()
        self.gpu = asyncio.Lock()
