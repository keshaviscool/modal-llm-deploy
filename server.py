import json
from typing import Union, List 
from pydantic import BaseModel, Field
from modal import fastapi_endpoint, concurrent

# Define the required things for building the server
from engine import app, HFEngine, HF_DOCKER_IMAGE
from constants import MIN_CONTAINERS, NUM_CONCURRENT_REQUESTS, TIMEOUT

class JobInput(BaseModel):
    messages: Union[str, List[dict]]
    max_new_tokens: int | None = Field(default=512)
    temperature: float | None = Field(default=0.4)
    top_p: float | None = Field(default=0.95)


@app.function(
    min_containers=MIN_CONTAINERS, # set as 1, instead of using KEEP WARM (depreceated)
    timeout=TIMEOUT, 
    image=HF_DOCKER_IMAGE
)
@concurrent(max_inputs=NUM_CONCURRENT_REQUESTS)
@fastapi_endpoint(method="POST", label="completion")
async def completion(item: JobInput):
    model = HFEngine()
    gen_kwargs = {
        "max_new_tokens": item.max_new_tokens,
        "temperature": item.temperature,
        "top_p": item.top_p,
        "do_sample": True
    }

    response = await model.inference.remote.aio(
        chat_input=item.messages, generation_kwargs=gen_kwargs
    )

    return {"response": response}