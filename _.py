import modal
from fastapi import FastAPI
from pydantic import BaseModel

app = modal.App("mistral-7b-instruct-api")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "uvicorn",
        "vllm",
        "huggingface_hub",
        "torch",
    )
)

web_app = FastAPI()

class Request(BaseModel):
    prompt: str
    max_tokens: int = 32000
    temperature: float = 0.1

@app.function(
    image=image,
    gpu="B200",
    scaledown_window=3000,
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_ID,
        dtype="auto",
        max_model_len=4096,
    )

    @web_app.post("/generate")
    async def generate(req: Request):

        formatted_prompt = f"[INST] {req.prompt} [/INST]"

        sampling_params = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stop=["</s>"],
        )

        outputs = llm.generate([formatted_prompt], sampling_params)
        response_text = outputs[0].outputs[0].text.strip()

        return {"response": response_text}

    return web_app