import io
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, UploadFile, Form, WebSocketDisconnect
from fastapi.responses import FileResponse
from model.model import Model

model = Model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the background GPU worker when FastAPI boots up
    model.start_worker()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def serve_frontend():
    """Serves the main HTML interface."""
    # This assumes index.html is in the exact same folder as rest.py
    return FileResponse("./view/index.html")

@app.post("/upload")
async def process_upload(file: UploadFile, text_prompt: str = Form(...)):
    """Handles standard image and label uploads."""
    frame_bytes = await file.read()

    # Assuming text_prompt comes in as a comma-separated string
    target_labels = [label.strip() for label in text_prompt.split(",")]

    try:
        results = await model.submit_frame(frame_bytes, target_labels)
        return {"status": "success", "data": results}
    except asyncio.CancelledError:
        # Catch the case where an upload is dropped by the buffer
        return {"status": "error", "message": "Server busy. Request dropped."}


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the live webcam feed."""
    await websocket.accept()

    try:
        labels_data = await websocket.receive_json()
        target_labels = labels_data.get("labels", [["a photo of a person"]])

        while True:
            # 1. Wait for the client to send the frame
            frame_bytes = await websocket.receive_bytes()

            try:
                # 2. Submit to the model and wait for the GPU worker to reply
                results = await model.submit_frame(frame_bytes, target_labels)

                # 3. Send formatted results back to trigger the next client frame
                await websocket.send_json({"status": "success", "data": results})

            except asyncio.CancelledError:
                # If this frame was overwritten in the buffer before processing, ignore it.
                # The client is stuck waiting for a response, so send a 'dropped' signal
                # to prompt the client to send the next frame anyway.
                await websocket.send_json({"status": "dropped"})

    except WebSocketDisconnect:
        print("Client disconnected.")