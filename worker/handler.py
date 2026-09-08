import base64
import glob
import io
import json
import os
import time

import runpod
import torch
from diffusers import DiffusionPipeline


MODEL_ID = os.environ["MODEL_ID"]
CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
IS_Z_IMAGE = MODEL_ID.lower().endswith("z-image-turbo")


def find_snapshot(model_id):
    repo_name = "models--" + model_id.replace("/", "--")
    snapshots = glob.glob(os.path.join(CACHE_ROOT, repo_name, "snapshots", "*"))
    if not snapshots:
        raise RuntimeError(f"Cached Model unavailable: {model_id}")
    return max(snapshots, key=os.path.getmtime)


MODEL_PATH = find_snapshot(MODEL_ID)
print(json.dumps({"event": "model_load_start", "model": MODEL_ID, "path": MODEL_PATH}))
load_started = time.perf_counter()
PIPE = DiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    low_cpu_mem_usage=not IS_Z_IMAGE,
)
PIPE.to("cuda")
PIPE.set_progress_bar_config(disable=True)
print(json.dumps({"event": "model_load_complete", "seconds": round(time.perf_counter() - load_started, 3)}))


def handler(job):
    request = job.get("input") or {}
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "input.prompt must be a non-empty string"}

    width = int(request.get("width", 1024))
    height = int(request.get("height", 1024))
    if not (256 <= width <= 2048 and 256 <= height <= 2048):
        return {"error": "width and height must be between 256 and 2048"}
    if width % 16 or height % 16:
        return {"error": "width and height must be divisible by 16"}

    steps = int(request.get("steps", 9 if IS_Z_IMAGE else 4))
    guidance = float(request.get("guidance_scale", 0.0 if IS_Z_IMAGE else 1.0))
    seed = int(request.get("seed", torch.seed() % (2**63 - 1)))
    generator = torch.Generator(device="cuda").manual_seed(seed)

    started = time.perf_counter()
    image = PIPE(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "status": "success",
        "model": MODEL_ID,
        "seed": seed,
        "width": image.width,
        "height": image.height,
        "generation_ms": round((time.perf_counter() - started) * 1000),
        "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
    }


runpod.serverless.start({"handler": handler})
