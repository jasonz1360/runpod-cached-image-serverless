# RunPod cached image workers

Two independent serverless images:

- `ghcr.io/jasonz1360/flux2-klein-4b-worker:v1.0.0`
- `ghcr.io/jasonz1360/z-image-turbo-worker:v1.0.0`

Each loads only the RunPod Cached Model selected for its endpoint. No model
weights are stored in this repository or baked into either container image.
