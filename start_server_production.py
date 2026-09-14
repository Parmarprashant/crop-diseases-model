import os
import uvicorn

if __name__ == "__main__":
    os.environ["CHECKPOINT_PATH"] = "weights/efficientnet_b5_cbam_best.pt"
    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, log_level="info")
