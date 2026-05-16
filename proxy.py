import json
import time
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="DS2API Browser Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DS2API_URL = "http://127.0.0.1:5001"
API_KEYS = ["sk-default", "sk-test123456"]
ADMIN_KEY = "admin"


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing API key")

    token = authorization.replace("Bearer ", "").strip()
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return token


@app.get("/v1/models")
async def list_models(authorization: str = Header(...)):
    verify_api_key(authorization)
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{DS2API_URL}/v1/models", headers={"Authorization": f"Bearer sk-test123456"})
        return resp.json()


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, authorization: str = Header(...)):
    verify_api_key(authorization)
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{DS2API_URL}/v1/models/{model_id}", headers={"Authorization": f"Bearer sk-test123456"})
        return resp.json()


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: str = Header(...),
):
    verify_api_key(authorization)

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    async with httpx.AsyncClient() as client:
        if request.stream:
            async def stream_with_cleanup():
                async with httpx.AsyncClient() as stream_client:
                    async with stream_client.stream(
                        "POST",
                        f"{DS2API_URL}/v1/chat/completions",
                        json=request.model_dump(),
                        headers={"Authorization": "Bearer sk-test123456"},
                        timeout=120,
                    ) as resp:
                        async for line in resp.aiter_lines():
                            yield line + "\n"

            return StreamingResponse(
                stream_with_cleanup(),
                media_type="text/event-stream",
            )

        resp = await client.post(
            f"{DS2API_URL}/v1/chat/completions",
            json=request.model_dump(),
            headers={"Authorization": "Bearer sk-test123456"},
            timeout=120,
        )
        return resp.json()


@app.get("/anthropic/v1/models")
async def anthropic_models(authorization: str = Header(...)):
    verify_api_key(authorization)

    return {
        "data": [
            {"id": "claude-sonnet-4-6", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "claude-opus-4-6", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "claude-haiku-4-5", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
        ],
        "object": "list",
    }


@app.post("/anthropic/v1/messages")
async def anthropic_messages(request: Request, authorization: str = Header(...)):
    verify_api_key(authorization)

    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "claude-sonnet-4-6")
    stream = body.get("stream", False)

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    prompt = messages[-1].get("content", "")

    async with httpx.AsyncClient() as client:
        if stream:
            async def stream_with_cleanup():
                async with httpx.AsyncClient() as stream_client:
                    async with stream_client.stream(
                        "POST",
                        f"{DS2API_URL}/v1/chat/completions",
                        json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}], "stream": True},
                        headers={"Authorization": "Bearer sk-test123456"},
                        timeout=120,
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    continue
                                try:
                                    data = json.loads(data_str)
                                    content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if content:
                                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
                                except json.JSONDecodeError:
                                    pass
                    
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            return StreamingResponse(
                stream_with_cleanup(),
                media_type="text/event-stream",
            )

        resp = await client.post(
            f"{DS2API_URL}/v1/chat/completions",
            json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}], "stream": False},
            headers={"Authorization": "Bearer sk-test123456"},
            timeout=120,
        )
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        return {
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": content}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": len(prompt.split()),
                "output_tokens": len(content.split()),
            },
        }


@app.post("/v1beta/models/{model}:generateContent")
async def gemini_generate(model: str, request: Request, authorization: str = Header(...)):
    verify_api_key(authorization)

    body = await request.json()
    contents = body.get("contents", [])

    if not contents:
        raise HTTPException(status_code=400, detail="No contents provided")

    prompt = contents[-1].get("parts", [{}])[0].get("text", "")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{DS2API_URL}/v1/chat/completions",
            json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}], "stream": False},
            headers={"Authorization": "Bearer sk-test123456"},
            timeout=120,
        )
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": content}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": len(prompt.split()),
                "candidatesTokenCount": len(content.split()),
                "totalTokenCount": len(prompt.split()) + len(content.split()),
            },
        }


@app.post("/v1beta/models/{model}:streamGenerateContent")
async def gemini_stream_generate(model: str, request: Request, authorization: str = Header(...)):
    verify_api_key(authorization)

    body = await request.json()
    contents = body.get("contents", [])

    if not contents:
        raise HTTPException(status_code=400, detail="No contents provided")

    prompt = contents[-1].get("parts", [{}])[0].get("text", "")

    async def stream_with_cleanup():
        async with httpx.AsyncClient() as stream_client:
            async with stream_client.stream(
                "POST",
                f"{DS2API_URL}/v1/chat/completions",
                json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}], "stream": True},
                headers={"Authorization": "Bearer sk-test123456"},
                timeout=120,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield f"data: {json.dumps({'candidates': [{'content': {'parts': [{'text': content}], 'role': 'model'}}]})}\n\n"
                        except json.JSONDecodeError:
                            pass
            
            yield f"data: {json.dumps({'candidates': [{'content': {'parts': [], 'role': 'model'}, 'finishReason': 'STOP'}], 'usageMetadata': {'promptTokenCount': 0, 'candidatesTokenCount': 0, 'totalTokenCount': 0}})}\n\n"

    return StreamingResponse(
        stream_with_cleanup(),
        media_type="text/event-stream",
    )


@app.get("/api/version")
async def ollama_version():
    return {"version": "0.1.0"}


@app.get("/api/tags")
async def ollama_tags():
    return {
        "models": [
            {"name": "deepseek-chat", "model": "deepseek-chat"},
            {"name": "deepseek-reasoner", "model": "deepseek-reasoner"},
        ]
    }


@app.post("/api/show")
async def ollama_show(request: Request):
    body = await request.json()
    model = body.get("model", "deepseek-chat")

    return {
        "id": model,
        "capabilities": ["tools", "thinking"],
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    return {"status": "ok", "accounts": {"total": 1, "in_use": 0, "available": 1}}


@app.get("/admin/stats")
async def admin_stats(admin_key: str = Header(...)):
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    return {"total": 1, "in_use": 0, "available": 1, "logged_in": 1, "queue_size": 0}


@app.get("/admin/config")
async def get_config(admin_key: str = Header(...)):
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    return {
        "server": {"host": "0.0.0.0", "port": 5002},
        "browser": {"headless": True, "max_concurrent_per_account": 1, "timeout": 60000},
        "default_proxy": None,
        "account_count": 1,
    }


def main():
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5002,
    )


if __name__ == "__main__":
    main()
