# Deploy Hugging Face Models on Modal

Deploy any Hugging Face causal language model as a streaming API endpoint using Modal.

## Project Structure

```
constants.py   - Model configuration (model name, GPU, timeouts)
engine.py      - Modal image build and HFEngine inference class
server.py      - FastAPI streaming endpoint
requirements.txt - Python dependencies installed in the Modal container
```

## Prerequisites

- Python 3.10+
- A [Modal](https://modal.com) account
- A [Hugging Face](https://huggingface.co) account with an access token

## Setup

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal setup
```

This will open a browser window to log in and link your Modal account.

### 3. Create a Hugging Face Secret on Modal

Generate an access token at https://huggingface.co/settings/tokens, then create a Modal secret:

```bash
modal secret create hfsecret HF_TOKEN=hf_your_token_here
```

This is required to download gated models (e.g., Mistral, Llama).

### 4. Configure the Model

Edit `constants.py` to set the model you want to deploy:

```python
MODEL_DIR = "/model"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # change to any HF causal LM

APP_NAME = f"{BASE_MODEL.lower()}-deployment"

MIN_CONTAINERS = 1
NUM_CONCURRENT_REQUESTS = 10
TIMEOUT = 600

GPU_CONFIG = "H100"  # options: "H100", "A100", "A10G", "L4", "T4", etc.
```

- `BASE_MODEL` -- the Hugging Face model ID.
- `GPU_CONFIG` -- the GPU type to use. Pick based on model size (e.g., H100/A100 for 7B+ models, A10G/L4 for smaller models).
- `TIMEOUT` -- idle timeout in seconds before the container scales down.
- `MIN_CONTAINERS` -- minimum containers kept warm (set to 0 to scale to zero).

### 5. Run the Server

Start in dev mode (hot-reloads on file changes):

```bash
modal serve server.py
```

Modal will print a URL for the endpoint. The endpoint accepts POST requests.

### 6. Deploy to Production

```bash
modal deploy server.py
```

## API Usage

Send a POST request to the endpoint URL printed by Modal:

```bash
curl -X POST https://your-modal-url.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is gravity?"}],
    "max_new_tokens": 512,
    "temperature": 0.4,
    "top_p": 0.95
  }'
```

### Request Body

| Field           | Type              | Default | Description                        |
|-----------------|-------------------|---------|------------------------------------|
| messages        | string or list    | --      | A string or list of chat messages  |
| max_new_tokens  | int               | 512     | Maximum tokens to generate         |
| temperature     | float             | 0.4     | Sampling temperature               |
| top_p           | float             | 0.95    | Top-p (nucleus) sampling threshold |

The response is a server-sent event (SSE) stream. Each event contains a JSON object with a `text` field holding the generated token.

## Notes

- The first cold start takes several minutes as the model is downloaded and the container image is built. Subsequent starts are fast due to caching.
- Set `MIN_CONTAINERS = 0` in `constants.py` to scale to zero when idle (saves cost, but adds cold start latency).
- The Hugging Face token is only needed for gated/private models. Public models work without it, but the secret must still exist in Modal.
