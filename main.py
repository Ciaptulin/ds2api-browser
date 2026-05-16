import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 自动加载项目目录下的 .env
load_dotenv(Path(__file__).parent / ".env")

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
manager = AccountManager(max_inflight=2)


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


@app.get("/")
async def admin_panel():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=ADMIN_HTML)


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


ADMIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DS2API · 终端</title>
<style>
:root{--bg:#080c12;--panel:#0f1620;--border:#1a2738;--text:#a8c0d8;--dim:#4a6078;--accent:#64d8ff;--green:#4ade80;--red:#f87171;--amber:#fbbf24}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'JetBrains Mono','Sarasa Mono SC','Source Code Pro','Cascadia Code',monospace;background:var(--bg);color:var(--text);min-height:100vh;font-size:13px;line-height:1.6;-webkit-font-smoothing:antialiased}
body::before{content:'';position:fixed;inset:0;background:
  radial-gradient(ellipse 80% 50% at 50% -20%,rgba(100,216,255,.04),transparent),
  linear-gradient(180deg,transparent 0%,rgba(100,216,255,.01) 50%,transparent 100%);
  pointer-events:none;z-index:0}

.topbar{position:sticky;top:0;z-index:10;background:var(--panel);border-bottom:1px solid var(--border);padding:14px 24px;display:flex;align-items:center;gap:12px}
.topbar .dot{width:7px;height:7px;background:var(--green);box-shadow:0 0 6px var(--green);animation:glow 2s ease-in-out infinite}
.topbar .title{font-weight:800;font-size:14px;color:var(--accent);letter-spacing:2px}
.topbar .tag{font-size:10px;color:var(--dim);letter-spacing:1px;border:1px solid var(--border);padding:3px 8px}
@keyframes glow{0%,100%{box-shadow:0 0 4px var(--green)}50%{box-shadow:0 0 14px var(--green)}}

.main{position:relative;z-index:1;max-width:960px;margin:0 auto;padding:28px 16px}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:28px}
@media(max-width:600px){.grid{grid-template-columns:repeat(2,1fr);gap:8px}}

.stat{background:var(--panel);border:1px solid var(--border);padding:18px 16px;position:relative;overflow:hidden}
.stat::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(100,216,255,.03) 0%,transparent 60%);pointer-events:none}
.stat .num{font-size:42px;font-weight:800;color:var(--accent);line-height:1}
.stat .label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:2px;margin-top:4px}

.card{background:var(--panel);border:1px solid var(--border);margin-bottom:16px}
.card-head{display:flex;align-items:center;justify-content:space-between;padding:14px 18px;border-bottom:1px solid var(--border);background:rgba(100,216,255,.02)}
.card-head h2{font-size:11px;color:var(--accent);letter-spacing:2px;font-weight:800}
.card-head .prompt{color:var(--dim);font-weight:800;margin-right:6px}
.card-body{padding:16px 18px}

table{width:100%;border-collapse:collapse}
thead{border-bottom:2px solid var(--border)}
th{padding:10px 8px;text-align:left;color:var(--dim);font-weight:700;font-size:10px;letter-spacing:1px;white-space:nowrap}
td{padding:9px 8px;border-bottom:1px solid rgba(26,39,56,.6);word-break:break-all;font-size:12px}
tr:hover td{background:rgba(100,216,255,.02)}
@media(max-width:600px){th{font-size:9px;padding:8px 4px}td{font-size:11px;padding:8px 4px}}

.badge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;font-size:10px;font-weight:700;letter-spacing:.5px;white-space:nowrap}
.badge::before{content:'';width:5px;height:5px;display:inline-block}
.badge-on{color:var(--green);border:1px solid rgba(74,222,128,.35)}.badge-on::before{background:var(--green)}
.badge-off{color:var(--red);border:1px solid rgba(248,113,113,.3)}.badge-off::before{background:var(--red)}
.badge-idle{color:var(--dim);border:1px solid var(--border)}.badge-idle::before{background:var(--dim)}

.btn{display:inline-flex;align-items:center;gap:6px;padding:9px 18px;border:1px solid var(--border);background:transparent;color:var(--text);cursor:pointer;font-family:inherit;font-size:11px;font-weight:700;letter-spacing:1px;transition:all .15s;white-space:nowrap}
.btn:hover{border-color:var(--accent);color:var(--accent);background:rgba(100,216,255,.04)}
.btn-accent{background:var(--accent);color:var(--bg);border-color:var(--accent);font-weight:800}
.btn-accent:hover{background:transparent;color:var(--accent)}
.btn-sm{padding:6px 12px;font-size:10px}
@media(max-width:600px){.btn{padding:7px 12px;font-size:10px}}

textarea{width:100%;background:var(--bg);border:1px solid var(--border);padding:12px;color:var(--text);font-family:inherit;font-size:12px;line-height:1.7;min-height:90px;resize:vertical}
textarea:focus{outline:none;border-color:var(--accent);box-shadow:inset 0 0 0 1px rgba(100,216,255,.1)}
textarea::placeholder{color:var(--dim)}

.help{font-size:10px;color:var(--dim);margin-bottom:10px;letter-spacing:.5px;opacity:.7}

.bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:10px}
.spacer{flex:1}

.toast{position:fixed;top:24px;right:24px;z-index:99;padding:12px 20px;font-size:12px;font-weight:700;letter-spacing:.5px;animation:slide .25s;backdrop-filter:blur(8px)}
.toast-ok{background:rgba(74,222,128,.12);color:var(--green);border:1px solid rgba(74,222,128,.25)}
.toast-err{background:rgba(248,113,113,.12);color:var(--red);border:1px solid rgba(248,113,113,.25)}
@keyframes slide{from{transform:translateY(-10px);opacity:0}to{transform:translateY(0);opacity:1}}

.empty{color:var(--dim);padding:32px 0;text-align:center;font-size:12px;letter-spacing:1px}
.hide-mobile{display:table-cell}
@media(max-width:600px){.hide-mobile{display:none}}

.pulse{animation:flicker 3s ease-in-out infinite}
@keyframes flicker{0%,100%{opacity:1}50%{opacity:.5}}

.ellipsis{max-width:180px;overflow:hidden;text-overflow:ellipsis;display:block}
</style>
</head>
<body>
<div class="topbar">
  <div class="dot"></div>
  <span class="title">DS2API</span>
  <div class="spacer" style="flex:1"></div>
  <span class="tag">▸ 浏览器模式</span>
</div>

<div class="main">
  <!-- 统计卡片 -->
  <div class="grid" id="stats">
    <div class="stat"><div class="num">—</div><div class="label">账号总数</div></div>
    <div class="stat"><div class="num">—</div><div class="label">使用中</div></div>
    <div class="stat"><div class="num">—</div><div class="label">可用</div></div>
    <div class="stat"><div class="num">—</div><div class="label">已登录</div></div>
    <div class="stat"><div class="num">—</div><div class="label">排队中</div></div>
  </div>

  <!-- 账号列表 -->
  <div class="card">
    <div class="card-head">
      <h2><span class="prompt">&gt;</span>账号列表</h2>
      <button class="btn btn-sm" onclick="loadAll()">刷新</button>
    </div>
    <div class="card-body">
      <table>
        <thead><tr><th>邮箱</th><th class="hide-mobile">备注</th><th>登录</th><th>状态</th><th class="hide-mobile">错误</th></tr></thead>
        <tbody id="tbl"><tr><td colspan="5" class="empty">正在加载…</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- 导入账号 -->
  <div class="card">
    <div class="card-head">
      <h2><span class="prompt">&gt;</span>导入账号</h2>
    </div>
    <div class="card-body">
      <div class="help">格式：邮箱:密码 ，每行一个账号</div>
      <textarea id="inp" placeholder="user@gmail.com:password"></textarea>
      <div class="bar">
        <button class="btn btn-accent" onclick="doImport()">▸ 导入</button>
        <span id="msg" style="font-size:11px;color:var(--dim)"></span>
      </div>
    </div>
  </div>
</div>

<script>
const H=location.origin
function toast(m,ok){
  const e=document.createElement('div')
  e.className='toast toast-'+(ok?'ok':'err')
  e.textContent=m
  document.body.appendChild(e)
  setTimeout(()=>e.remove(),2800)
}
async function api(p,o={}){
  const hd={};if(o.json)hd['Content-Type']='application/json'
  Object.assign(hd,o.headers||{})
  const r=await fetch(H+p,{headers:hd,method:o.method||'GET',body:o.body})
  if(!r.ok){const e=await r.text();throw new Error(e||r.status)}
  return r.json()
}
async function loadAll(){
  try{
    const s=await api('/readyz')
    document.getElementById('stats').innerHTML=
      `<div class="stat"><div class="num">${s.accounts.total}</div><div class="label">账号总数</div></div>
       <div class="stat"><div class="num">${s.accounts.in_use}</div><div class="label">使用中</div></div>
       <div class="stat"><div class="num">${s.accounts.available}</div><div class="label">可用</div></div>
       <div class="stat"><div class="num">${s.accounts.logged_in}</div><div class="label">已登录</div></div>
       <div class="stat"><div class="num">${s.accounts.queue_size}</div><div class="label">排队中</div></div>`
  }catch(e){}
  try{
    const d=await api('/admin/accounts',{headers:{'admin-key':'admin'}})
    let r=''
    for(const a of d.accounts){
      r+=`<tr>
        <td><span class="ellipsis">${a.email}</span></td>
        <td class="hide-mobile">${a.name||'—'}</td>
        <td><span class="badge ${a.logged_in?'badge-on':'badge-off'}">${a.logged_in?'已登录':'未登录'}</span></td>
        <td><span class="badge ${a.in_use?'badge-on':'badge-idle'}">${a.in_use?'使用中':'空闲'}</span></td>
        <td class="hide-mobile">${a.error_count>0?'<span class="badge badge-off">'+a.error_count+'次</span>':'—'}</td>
      </tr>`
    }
    document.getElementById('tbl').innerHTML=r||'<tr><td colspan="5" class="empty">暂无账号</td></tr>'
  }catch(e){
    document.getElementById('tbl').innerHTML='<tr><td colspan="5" style="color:var(--red)">加载失败：'+e.message+'</td></tr>'
  }
}
async function doImport(){
  const v=document.getElementById('inp').value.trim()
  if(!v)return toast('请先输入账号信息',0)
  const accts=[]
  for(const l of v.split('\\n')){
    const t=l.trim();if(!t)continue
    const p=t.split(':',3)
    if(p.length>=2)accts.push({email:p[0].trim(),password:p[1],name:p[2]||''})
  }
  if(!accts.length)return toast('格式错误，请用 邮箱:密码 格式',0)
  try{
    const d=await api('/admin/accounts/import',{
      method:'POST',json:true,
      body:JSON.stringify({accounts:accts}),
      headers:{'admin-key':'admin'}
    })
    document.getElementById('inp').value=''
    document.getElementById('msg').textContent='成功导入 '+d.imported+' 个账号'
    toast('已导入 '+d.imported+' / '+accts.length+' 个',1)
    loadAll()
  }catch(e){toast('导入失败：'+e.message,0)}
}
loadAll()
setInterval(loadAll,15000)
</script>
</body>
</html>"""


def main():
    import uvicorn

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
    )


if __name__ == "__main__":
    main()
