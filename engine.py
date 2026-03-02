import os
from typing import List, Union

# modal imports
from modal import Image, App, gpu, method, enter, exit, concurrent, Secret

# constants
from constants import *


app = App(name=APP_NAME)

def download_model_to_folder():
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt"],  # Using safetensors
    )

HF_DOCKER_IMAGE = (
    Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    .pip_install_from_requirements("./requirements.txt")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file("constants.py", "/root/constants.py", copy=True)
    .run_function(download_model_to_folder, secrets=[Secret.from_name("hfsecret")])
    .add_local_python_source("engine", "server")
)


@app.cls(
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    scaledown_window=TIMEOUT,
    secrets=[Secret.from_name("hfsecret")],
    image=HF_DOCKER_IMAGE,
)
@concurrent(max_inputs=NUM_CONCURRENT_REQUESTS)
class HFEngine:
    model_name_or_path: str = MODEL_DIR
    device: str = "cuda"

    @enter()
    def start_engine(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        return self
    
    @exit()
    def terminate_engine(self):
        import gc
        import torch

        del self.model
        torch.cuda.synchronize()
        gc.collect()

    @method()
    def inference(self, chat_input: Union[str, List[dict]], generation_kwargs: dict) -> str:
        if isinstance(chat_input, str):
            chat_input = [{"role": "user", "content": chat_input}]
        input_ids = self.tokenizer.apply_chat_template(
            conversation=chat_input, tokenize=True, return_tensors="pt", return_dict=False
        ).to(self.device)

        output = self.model.generate(
            input_ids=input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            **generation_kwargs
        )

        # decode only the newly generated tokens
        response = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response