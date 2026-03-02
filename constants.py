MODEL_DIR = "/model"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2" # example: mistralai/Mistral-7B-Instruct-v0.2

APP_NAME = f"{BASE_MODEL.lower().replace('/', '-')}-deployment" # should be in lower case, no slashes
APP_NAME = APP_NAME[:60]

MIN_CONTAINERS = 1

NUM_CONCURRENT_REQUESTS = 10

# timeout: This is the server timeout after which it would be shutdown the server. 
TIMEOUT = 600

GPU_CONFIG = "H100"