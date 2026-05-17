import asyncio
import collections
import json
import logging
import logging.handlers
import os
import time
import uuid
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_BUFFER_SIZE = 500


class MemoryLogHandler(logging.Handler):
    """Ring buffer handler that keeps recent log records in memory."""
    def __init__(self, capacity=LOG_BUFFER_SIZE):
        super().__init__()
        self.buffer = collections.deque(maxlen=capacity)

    def emit(self, record):
        self.buffer.append(self.format(record))

    def get_logs(self, n=100):
        return list(self.buffer)[-n:]

    def clear(self):
        self.buffer.clear()


_mem_handler = MemoryLogHandler(LOG_BUFFER_SIZE)
_mem_handler.setFormatter(logging.Formatter(LOG_FORMAT))

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)
logging.getLogger().addHandler(_mem_handler)
logger = logging.getLogger(__name__)

_file_handler: logging.Handler | None = None

from dotenv import load_dotenv

# 自动加载项目目录下的 .env
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from account_manager import AccountManager
from config import Config, load_config

app = FastAPI(title="DS2API Browser")

# 挂载静态文件
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config: Config = load_config()
manager = AccountManager(
    max_active_browsers=int(os.getenv("DS2API_MAX_ACTIVE_BROWSERS", "50")),
)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[list[dict]] = None


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
            {"id": "deepseek-v4-flash", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
            {"id": "deepseek-v4-pro", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
        ],
        "object": "list",
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, authorization: str = Header(...)):
    verify_api_key(authorization)

    models = {
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
    
    if request.tools:
        tool_desc = json.dumps(request.tools, ensure_ascii=False)
        prompt += f"\n\n[SYSTEM INSTRUCTION: You have access to the following tools:\n{tool_desc}\nIf you must use a tool to fulfill the request, output ONLY a JSON block wrapped in <tool_call>...</tool_call> tags, like:\n<tool_call>{{\"name\": \"tool_name\", \"arguments\": {{\"arg1\": \"value\"}} }}</tool_call>\nDo NOT output any other text if you are calling a tool.]"

    model = request.model

    account = await manager.acquire()

    try:
        browser = await manager.get_or_create_browser_with_retry(account, headless=config.browser.headless)

        if request.stream:
            async def stream_with_cleanup():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                try:
                    is_tool_call = False
                    not_tool_call = False
                    content_buffer = ""
                    
                    async for chunk_data in browser.stream_message(prompt, timeout=120, model=model):
                        chunk_type = chunk_data.get("type", "content")
                        chunk_text = chunk_data.get("chunk", "")
                        
                        if chunk_type == "thinking":
                            delta = {"reasoning_content": chunk_text}
                        else:
                            if request.tools and not is_tool_call and not not_tool_call:
                                content_buffer += chunk_text
                                # Wait until we have enough characters to decide
                                if len(content_buffer) < 12:
                                    if not "<tool_call>".startswith(content_buffer):
                                        not_tool_call = True
                                        delta = {"content": content_buffer}
                                    else:
                                        continue # keep buffering
                                else:
                                    if content_buffer.startswith("<tool_call>"):
                                        is_tool_call = True
                                        continue # buffer the whole tool call
                                    else:
                                        not_tool_call = True
                                        delta = {"content": content_buffer}
                            elif request.tools and is_tool_call:
                                content_buffer += chunk_text
                                continue # buffer until stream ends
                            else:
                                delta = {"content": chunk_text}

                        data = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    
                    if is_tool_call:
                        # Process buffered tool call at the end
                        import re
                        m = re.search(r'<tool_call>(.*?)</tool_call>', content_buffer, re.DOTALL)
                        if m:
                            try:
                                tcall = json.loads(m.group(1))
                                t_name = tcall.get("name", "")
                                t_args = json.dumps(tcall.get("arguments", {}))
                                delta = {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": f"call_{uuid.uuid4().hex[:8]}",
                                            "type": "function",
                                            "function": {
                                                "name": t_name,
                                                "arguments": t_args
                                            }
                                        }
                                    ]
                                }
                                data = {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": request.model,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": "tool_calls"}]
                                }
                                yield f"data: {json.dumps(data)}\n\n"
                            except Exception as e:
                                logger.error("Failed to parse tool call: %s", e)
                    
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

        response_data = await browser.send_message(prompt, timeout=120, model=model)
        
        await manager.release(account)

        # Token counts are estimated by word splitting; not exact tokenization
        content = response_data.get("content", "")
        reasoning_content = response_data.get("reasoning_content", "")
        
        prompt_tokens = len(prompt.split())
        completion_tokens = len(content.split()) + len(reasoning_content.split())

        message_data = {"role": "assistant", "content": content}
        if reasoning_content:
            message_data["reasoning_content"] = reasoning_content
            
        finish_reason = "stop"
        
        if request.tools and "<tool_call>" in content:
            import re
            m = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
            if m:
                try:
                    tcall = json.loads(m.group(1))
                    message_data["content"] = None
                    message_data["tool_calls"] = [
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tcall.get("name", ""),
                                "arguments": json.dumps(tcall.get("arguments", {}))
                            }
                        }
                    ]
                    finish_reason = "tool_calls"
                except Exception as e:
                    logger.error("Failed to parse non-stream tool call: %s", e)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message_data,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    except Exception as e:
        await manager.mark_error(account)
        logger.error("Chat completion error for model=%s: %s", request.model, e)
        raise HTTPException(status_code=503, detail=str(e))



@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    stats = manager.get_stats()
    return {
        "status": "ok",
        "accounts": {
            "total": stats["total"],
            "in_use": stats["in_use"],
            "available": stats["available"],
            "logged_in": stats["logged_in"],
            "muted": stats["muted"],
            "queue_size": stats["queue_size"],
        },
    }


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
    new_accounts = []
    for acc in accounts:
        email = acc.get("email")
        password = acc.get("password")
        name = acc.get("name", "")
        proxy = acc.get("proxy")

        if email and password:
            if email not in manager.accounts:
                manager.add_account(email, password, name, proxy)
                new_accounts.append(manager.accounts[email])
                imported += 1

    # 持久化到 settings.json
    if imported > 0:
        saved = _load_settings()
        saved_accounts = saved.get("accounts", [])
        acc_map = {a.get("email"): a for a in saved_accounts if a.get("email")}
        for acc in accounts:
            e = acc.get("email")
            if e and acc.get("password"):
                acc_map[e] = acc
        saved["accounts"] = list(acc_map.values())
        _save_settings(saved)

    # 异步触发新导入账号的预登录 (只预热最多 max_active_browsers 个，防止卡死)
    async def prelogin_new_accounts():
        for i, account in enumerate(new_accounts):
            if i >= manager.max_active_browsers:
                break
            try:
                logger.info("Pre-logging in newly imported account %s...", account.email)
                await manager.get_or_create_browser_with_retry(
                    account, headless=config.browser.headless
                )
                logger.info("Pre-login OK: %s", account.email)
            except Exception as e:
                logger.error("Pre-login FAILED for %s: %s", account.email, e)

    if new_accounts:
        asyncio.create_task(prelogin_new_accounts())

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
            "is_muted": acc.is_muted,
            "muted_until": acc.muted_until,
            "error_count": acc.error_count,
        })

    return {"accounts": accounts, "total": len(accounts)}


@app.post("/admin/accounts/login")
async def login_account(request: Request, admin_key: str = Header(...)):
    """Manually trigger a login or reconnect for a specific account."""
    if admin_key != config.server.admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    body = await request.json()
    email = body.get("email")

    if not email or email not in manager.accounts:
        raise HTTPException(status_code=404, detail="Account not found")

    account = manager.accounts[email]

    async def _do_login():
        try:
            logger.info("Manual login triggered for %s...", email)
            # If it's already logged in, we might want to restart the browser.
            # get_or_create_browser_with_retry will reuse if account.browser exists.
            # To force a reconnect, we close the existing one first.
            if account.browser:
                try:
                    await account.browser.close()
                except Exception:
                    pass
                account.browser = None
                account.logged_in = False

            await manager.get_or_create_browser_with_retry(
                account, headless=config.browser.headless
            )
            logger.info("Manual login OK: %s", email)
        except Exception as e:
            logger.error("Manual login FAILED for %s: %s", email, e)

    asyncio.create_task(_do_login())
    return {"ok": True, "message": "Login task started"}


@app.post("/admin/verify")
async def admin_verify(request: Request):
    """Verify admin key for panel login."""
    body = await request.json()
    key = body.get("key", "")
    if key != config.server.admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return {"ok": True}


@app.post("/admin/chat")
async def admin_chat(request: Request, admin_key: str = Header(...)):
    """Chat endpoint for panel testing — uses admin key auth, no API key needed."""
    if admin_key != config.server.admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    body = await request.json()
    req = ChatCompletionRequest(**body)

    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    prompt = req.messages[-1].content
    
    if req.tools:
        tool_desc = json.dumps(req.tools, ensure_ascii=False)
        prompt += f"\n\n[SYSTEM INSTRUCTION: You have access to the following tools:\n{tool_desc}\nIf you must use a tool to fulfill the request, output ONLY a JSON block wrapped in <tool_call>...</tool_call> tags, like:\n<tool_call>{{\"name\": \"tool_name\", \"arguments\": {{\"arg1\": \"value\"}} }}</tool_call>\nDo NOT output any other text if you are calling a tool.]"
        
    model = req.model
    account = await manager.acquire()

    try:
        browser = await manager.get_or_create_browser_with_retry(account, headless=config.browser.headless)

        if req.stream:
            async def stream_with_cleanup():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                try:
                    is_tool_call = False
                    not_tool_call = False
                    content_buffer = ""
                    
                    async for chunk_data in browser.stream_message(prompt, timeout=120, model=model):
                        chunk_type = chunk_data.get("type", "content")
                        chunk_text = chunk_data.get("chunk", "")
                        
                        if chunk_type == "thinking":
                            delta = {"reasoning_content": chunk_text}
                        else:
                            if req.tools and not is_tool_call and not not_tool_call:
                                content_buffer += chunk_text
                                if len(content_buffer) < 12:
                                    if not "<tool_call>".startswith(content_buffer):
                                        not_tool_call = True
                                        delta = {"content": content_buffer}
                                    else:
                                        continue
                                else:
                                    if content_buffer.startswith("<tool_call>"):
                                        is_tool_call = True
                                        continue
                                    else:
                                        not_tool_call = True
                                        delta = {"content": content_buffer}
                            elif req.tools and is_tool_call:
                                content_buffer += chunk_text
                                continue
                            else:
                                delta = {"content": chunk_text}
                        
                        data = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": req.model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        
                    if is_tool_call:
                        import re
                        m = re.search(r'<tool_call>(.*?)</tool_call>', content_buffer, re.DOTALL)
                        if m:
                            try:
                                tcall = json.loads(m.group(1))
                                t_name = tcall.get("name", "")
                                t_args = json.dumps(tcall.get("arguments", {}))
                                delta = {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": f"call_{uuid.uuid4().hex[:8]}",
                                            "type": "function",
                                            "function": {
                                                "name": t_name,
                                                "arguments": t_args
                                            }
                                        }
                                    ]
                                }
                                data = {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": req.model,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": "tool_calls"}]
                                }
                                yield f"data: {json.dumps(data)}\n\n"
                            except Exception as e:
                                logger.error("Failed to parse admin stream tool call: %s", e)
                                
                    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
                finally:
                    await manager.release(account)

            return StreamingResponse(stream_with_cleanup(), media_type="text/event-stream")

        response_data = await browser.send_message(prompt, timeout=120, model=model)
        await manager.release(account)

        content = response_data.get("content", "")
        reasoning_content = response_data.get("reasoning_content", "")

        prompt_tokens = len(prompt.split())
        completion_tokens = len(content.split()) + len(reasoning_content.split())
        
        message_data = {"role": "assistant", "content": content}
        if reasoning_content:
            message_data["reasoning_content"] = reasoning_content
            
        finish_reason = "stop"
        
        if req.tools and "<tool_call>" in content:
            import re
            m = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
            if m:
                try:
                    tcall = json.loads(m.group(1))
                    message_data["content"] = None
                    message_data["tool_calls"] = [
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tcall.get("name", ""),
                                "arguments": json.dumps(tcall.get("arguments", {}))
                            }
                        }
                    ]
                    finish_reason = "tool_calls"
                except Exception as e:
                    logger.error("Failed to parse admin non-stream tool call: %s", e)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": message_data, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
        }
    except Exception as e:
        await manager.mark_error(account)
        logger.error("Admin chat error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))


SETTINGS_FILE = Path(__file__).parent / "settings.json"


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_settings(data: dict):
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _apply_settings(data: dict):
    """Apply settings to running config."""
    if "api_keys" in data:
        config.api_keys = [k.strip() for k in data["api_keys"] if k.strip()]
    if "admin_key" in data and data["admin_key"]:
        config.server.admin_key = data["admin_key"]
    if "log_file_enabled" in data:
        _setup_file_handler(
            enabled=data["log_file_enabled"],
            max_mb=data.get("log_file_max_mb", 10),
        )
    if "max_active_browsers" in data:
        manager.max_active_browsers = max(1, int(data["max_active_browsers"]))
    if "accounts" in data:
        for acc in data["accounts"]:
            if acc.get("email") and acc.get("password") and acc.get("email") not in manager.accounts:
                manager.add_account(
                    email=acc["email"],
                    password=acc["password"],
                    name=acc.get("name", ""),
                    proxy=acc.get("proxy")
                )


def _setup_file_handler(enabled: bool, max_mb: int = 10):
    """Add or remove a rotating file handler."""
    global _file_handler
    root = logging.getLogger()
    if _file_handler:
        root.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None
    if enabled:
        log_path = Path(__file__).parent / "ds2api.log"
        _file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_mb * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        _file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(_file_handler)
        logger.info("Log file enabled: %s (max %dMB)", log_path, max_mb)


@app.get("/admin/settings")
async def get_settings(admin_key: str = Header(...)):
    if admin_key != config.server.admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return {
        "api_keys": config.api_keys,
        "admin_key": config.server.admin_key,
        "headless": config.browser.headless,
        "port": config.server.port,
        "log_level": logging.getLogger().level,
        "log_file_enabled": _file_handler is not None,
        "log_file_max_mb": _load_settings().get("log_file_max_mb", 10),
        "max_active_browsers": manager.max_active_browsers,
    }


@app.post("/admin/settings")
async def save_settings(request: Request, admin_key: str = Header(...)):
    if admin_key != config.server.admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    body = await request.json()
    _save_settings(body)
    _apply_settings(body)
    return {"ok": True}


@app.get("/admin/logs")
async def get_logs(admin_key: str = Header(...), n: int = 100):
    if admin_key != config.server.admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return {"logs": _mem_handler.get_logs(n)}


@app.post("/admin/logs/clear")
async def clear_logs(admin_key: str = Header(...)):
    if admin_key != config.server.admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    _mem_handler.clear()
    return {"ok": True}


@app.post("/admin/logs/level")
async def set_log_level(request: Request, admin_key: str = Header(...)):
    if admin_key != config.server.admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    body = await request.json()
    level_name = body.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.getLogger().setLevel(level)
    logger.info("Log level changed to %s", level_name)
    return {"ok": True, "level": level_name}


@app.get("/")
async def admin_panel():
    return RedirectResponse(url="/static/index.html")


@app.on_event("startup")
async def startup():
    for acc in config.accounts:
        manager.add_account(
            email=acc.email,
            password=acc.password,
            name=acc.name,
            proxy=acc.proxy,
        )

    logger.info("Loaded %d accounts", len(config.accounts))

    # Apply persisted settings
    saved = _load_settings()
    if saved:
        _apply_settings(saved)
        logger.info("Applied persisted settings from settings.json")

    # Pre-login all accounts in background so they show online immediately
    asyncio.create_task(_prelogin_all())


async def _prelogin_all():
    """Pre-login a limited number of accounts at startup for instant readiness."""
    count = 0
    for email, account in manager.accounts.items():
        if count >= manager.max_active_browsers:
            break
        try:
            logger.info("Pre-logging in %s...", email)
            await manager.get_or_create_browser_with_retry(
                account, headless=config.browser.headless
            )
            logger.info("Pre-login OK: %s (muted=%s)", email, account.is_muted)
            count += 1
        except Exception as e:
            logger.error("Pre-login FAILED for %s: %s", email, e)


def main():
    import uvicorn

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
    )


if __name__ == "__main__":
    main()

