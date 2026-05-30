import asyncio
from typing import Optional, Tuple
import PIL.Image


class FrameBuffer:
    def __init__(self):
        self._new_frame_event = asyncio.Event()
        self._latest_data = None

    async def put(self, image, text_labels, future: asyncio.Future):
        # CRITICAL: If there is an unprocessed frame, cancel its Future so the caller doesn't hang
        if self._latest_data is not None:
            _, _, old_future = self._latest_data
            if not old_future.done():
                old_future.cancel()

        self._latest_data = (image, text_labels, future)
        self._new_frame_event.set()

    async def get(self) -> Tuple[PIL.Image.Image, list[str]]:
        """Wait for and retrieve the most recent frame."""
        await self._new_frame_event.wait()

        # Extract data and reset the event for the next frame
        data = self._latest_data
        self._new_frame_event.clear()
        return data