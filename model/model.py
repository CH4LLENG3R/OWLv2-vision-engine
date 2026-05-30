import io
import asyncio
import logging
from typing import List, Union

import PIL.Image
from common.singleton import SingletonMeta
from model.owlv2 import OWLv2

logger = logging.getLogger(__name__)


class Model(metaclass=SingletonMeta):
    def __init__(self):
        self._model = OWLv2()
        # Replace the custom FrameBuffer with a standard async Queue
        self._queue = asyncio.Queue()
        self._worker_task = None

    def start_worker(self):
        """Starts the background GPU worker."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._gpu_worker())

    async def _gpu_worker(self):
        """Processes frames sequentially from all connected users."""
        try:
            while True:
                # Pull requests from the queue in the exact order they arrived
                image, text_labels, future = await self._queue.get()

                # Optimization: If the user disconnected while waiting in line, 
                # their Future will be cancelled. Skip GPU processing to save resources.
                if future.cancelled():
                    self._queue.task_done()
                    continue

                try:
                    # Run the heavy PyTorch model
                    results = await asyncio.to_thread(self._model.predict, image, text_labels)

                    # Return the result to the specific user who requested it
                    if not future.done():
                        future.set_result(results)
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    if not future.done():
                        future.set_exception(e)
                finally:
                    # Mark this task as complete in the queue
                    self._queue.task_done()

        except asyncio.CancelledError:
            logger.info("GPU worker shut down cleanly.")

    def image_from_bytes(self, frame_bytes: bytes) -> PIL.Image.Image:
        return PIL.Image.open(io.BytesIO(frame_bytes)).convert("RGB")

    async def submit_frame(self, frame_bytes: bytes, target_labels: Union[str, List[str]]):
        """Submits a frame to the GPU queue and awaits the specific result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        image = self.image_from_bytes(frame_bytes)

        # Place the image, labels, and tracking future at the back of the line
        await self._queue.put((image, target_labels, future))

        # Suspend this user's WebSocket loop until the GPU gets to their frame
        return await future