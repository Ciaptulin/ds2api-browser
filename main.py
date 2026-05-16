import asyncio
import json
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from account_manager import AccountManager
from config import Config, load_config

app = FastAPI(title="DS2API Browser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config: Config = load_config()
manager = AccountManager(max_inflight=1)


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
    if token not in config.api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return token


@app.get("/v1/models")
async def list_models(authorization: str = Header(...)):
    verify_api_key(authorization)

    return {
        "data": [
            {"id": "deepseek-chat", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
            {"id": "deepseek-reasoner", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
            {"id": "deepseek-v4-flash", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
            {"id": "deepseek-v4-pro", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
            {"id": "deepseek-v4-flash-search", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
            {"id": "deepseek-v4-pro-search", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
            {"id": "deepseek-v4-vision", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
            {"id": "gpt-4o", "object": "model", "created": int(time.time()), "owned_by": "openai"},
            {"id": "gpt-4-turbo", "object": "model", "created": int(time.time()), "owned_by": "openai"},
            {"id": "claude-3-opus", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "claude-3-sonnet", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "gemini-pro", "object": "model", "created": int(time.time()), "owned_by": "google"},
        ],
        "object": "list",
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, authorization: str = Header(...)):
    verify_api_key(authorization)

    models = {
        "deepseek-chat": {"id": "deepseek-chat", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
        "deepseek-reasoner": {"id": "deepseek-reasoner", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
        "deepseek-v4-flash": {"id": "deepseek-v4-flash", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
        "deepseek-v4-pro": {"id": "deepseek-v4-pro", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
    }

    if model_id in models:
        return models[model_id]

    raise HTTPException(status_code=404, detail="Model not found")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: str = Header(...),
):
    verify_api_key(authorization)

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    prompt = request.messages[-1].content

    account = await manager.acquire()

    try:
        browser = await manager.get_or_create_browser_with_retry(account, headless=config.browser.headless)

        if request.stream:
            async def stream_with_cleanup():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                try:
                    async for chunk in browser.stream_message(prompt, timeout=120, model=request.model):
                        data = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    
                    final_data = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
                finally:
                    await manager.release(account)

            return StreamingResponse(
                stream_with_cleanup(),
                media_type="text/event-stream",
            )

        response_text = await browser.send_message(prompt, timeout=120, model=request.model)

        await manager.release(account)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split()),
            },
        }

    except Exception as e:
        await manager.mark_error(account)
        raise HTTPException(status_code=503, detail=str(e))


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

    account = await manager.acquire()

    try:
        browser = await manager.get_or_create_browser_with_retry(account, headless=config.browser.headless)

        if stream:
            async def stream_with_cleanup():
                try:
                    async for chunk in browser.stream_message(prompt, timeout=120, model=model):
                        data = {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": chunk},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"
                    
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'server_error', 'message': str(e)}})}\n\n"
                finally:
                    await manager.release(account)

            return StreamingResponse(
                stream_with_cleanup(),
                media_type="text/event-stream",
            )

        response_text = await browser.send_message(prompt, timeout=120, model=model)

        await manager.release(account)

        return {
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": response_text}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": len(prompt.split()),
                "output_tokens": len(response_text.split()),
            },
        }

    except Exception as e:
        await manager.mark_error(account)
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/v1beta/models/{model}:generateContent")
async def gemini_generate(model: str, request: Request, authorization: str = Header(...)):
    verify_api_key(authorization)

    body = await request.json()
    contents = body.get("contents", [])

    if not contents:
        raise HTTPException(status_code=400, detail="No contents provided")

    prompt = contents[-1].get("parts", [{}])[0].get("text", "")

    account = await manager.acquire()

    try:
        browser = await manager.get_or_create_browser_with_retry(account, headless=config.browser.headless)

        response_text = await browser.send_message(prompt, timeout=120, model=model)

        await manager.release(account)

        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": response_text}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": len(prompt.split()),
                "candidatesTokenCount": len(response_text.split()),
                "totalTokenCount": len(prompt.split()) + len(response_text.split()),
            },
        }

    except Exception as e:
        await manager.mark_error(account)
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/v1beta/models/{model}:streamGenerateContent")
async def gemini_stream_generate(model: str, request: Request, authorization: str = Header(...)):
    verify_api_key(authorization)

    body = await request.json()
    contents = body.get("contents", [])

    if not contents:
        raise HTTPException(status_code=400, detail="No contents provided")

    prompt = contents[-1].get("parts", [{}])[0].get("text", "")

    account = await manager.acquire()

    try:
        browser = await manager.get_or_create_browser_with_retry(account, headless=config.browser.headless)

        async def stream_with_cleanup():
            try:
                async for chunk in browser.stream_message(prompt, timeout=120, model=model):
                    data = {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": chunk}],
                                    "role": "model",
                                },
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                
                final_data = {
                    "candidates": [
                        {
                            "content": {"parts": [], "role": "model"},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 0,
                        "candidatesTokenCount": 0,
                        "totalTokenCount": 0,
                    },
                }
                yield f"data: {json.dumps(final_data)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
            finally:
                await manager.release(account)

        return StreamingResponse(
            stream_with_cleanup(),
            media_type="text/event-stream",
        )

    except Exception as e:
        await manager.mark_error(account)
        raise HTTPException(status_code=503, detail=str(e))


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
    stats = manager.get_stats()
    return {"status": "ok", "accounts": stats}


@app.get("/admin/stats")
async def admin_stats(admin_key: str = Header(...)):
    if admin_key != config.server.admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    return manager.get_stats()


@app.post("/admin/accounts/import")
async def import_accounts(request: Request, admin_key: str = Header(...)):
    if admin_key != config.server.admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    body = await request.json()
    accounts = body.get("accounts", [])

    if not accounts:
        raise HTTPException(status_code=400, detail="No accounts provided")

    imported = 0
    for acc in accounts:
        email = acc.get("email")
        password = acc.get("password")
        name = acc.get("name", "")
        proxy = acc.get("proxy")

        if email and password:
            manager.add_account(email, password, name, proxy)
            imported += 1

    return {"success": True, "imported": imported, "total": len(manager.accounts)}


@app.get("/admin/accounts")
async def list_accounts(admin_key: str = Header(...)):
    if admin_key != config.server.admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    accounts = []
    for email, acc in manager.accounts.items():
        accounts.append({
            "email": email,
            "name": acc.name,
            "in_use": acc.in_use,
            "logged_in": acc.logged_in,
            "error_count": acc.error_count,
        })

    return {"accounts": accounts, "total": len(accounts)}


@app.on_event("startup")
async def startup():
    for acc in config.accounts:
        manager.add_account(
            email=acc.email,
            password=acc.password,
            name=acc.name,
            proxy=acc.proxy,
        )

    print(f"Loaded {len(config.accounts)} accounts")


def main():
    import uvicorn

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
    )


if __name__ == "__main__":
    main()
