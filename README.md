# LLM Instruct Microservice

## Installation

1. Install docker
https://docs.docker.com/engine/install/ubuntu/
2. Clone repo
```bash
git clone https://github.com/DenisDiachkov/llm_instruct_ms
cd llm_instruct_ms
```
3. Build 
```bash
docker build -f dockerfile.prod -t llm_instruct_ms .
```
4. Run
```bash
docker compose up
```

## Inference
### GUI
Access GUI by ```127.0.0.1:8000```

![image](docs/gui.png)
### HTTP Request
#### Endpoint
GET 127.0.0.1:8000/generate
#### Parameters
```
prompt: str
system_prompt: str
reply_prefix: str
image: str  - base64 format
max_new_tokens: int
temperature: float
top_p: float
repetition_penalty: float
frequency_penalty: float
presence_penalty: float
```
#### Response
The server will respond with a JSON object containing:
```python
status: str - "error" or "finished"
content: str - LLMs response
```

### WebSocket
#### Send parameters
```python
prompt: str
system_prompt: str
reply_prefix: str
image: str  - base64 format
max_new_tokens: int
temperature: float
top_p: float
repetition_penalty: float
frequency_penalty: float
presence_penalty: float
```
#### Receive JSON
The server will send a JSON object containing:
```python
status: str - "error" or "finished"
content: str - LLMs streaming response
```
Can contain also other entries, depending on the status. See more here:  [asyncio_queue_manager](https://github.com/AGISwarm/asyncio-queue-manager/blob/dev/src/AGISwarm/asyncio_queue_manager/core.py)