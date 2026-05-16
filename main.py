import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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
            {"id": "deepseek-flash", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
            {"id": "deepseek-pro", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
        ],
        "object": "list",
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, authorization: str = Header(...)):
    verify_api_key(authorization)

    models = {
        "deepseek-flash": {"id": "deepseek-flash", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
        "deepseek-pro": {"id": "deepseek-pro", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
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

    model = request.model

    account = await manager.acquire()

    try:
        browser = await manager.get_or_create_browser_with_retry(account, headless=config.browser.headless)

        if request.stream:
            async def stream_with_cleanup():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                try:
                    async for chunk in browser.stream_message(prompt, timeout=120, model=model):
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

        response_text = await browser.send_message(prompt, timeout=120, model=model)

        await manager.release(account)

        # Token counts are estimated by word splitting; not exact tokenization
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response_text.split())

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
            "is_muted": acc.is_muted,
            "muted_until": acc.muted_until,
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

    logger.info("Loaded %d accounts", len(config.accounts))


ADMIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DS2API · 控制台</title>
<style>
:root{--bg:#060b10;--panel:#0b1219;--border:#15202e;--text:#9bb5cf;--dim:#3d5268;--accent:#5cc8ff;--green:#3fb950;--red:#f85149;--amber:#d29922;--row-hover:rgba(92,200,255,.03)}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'JetBrains Mono','Sarasa Mono SC','Cascadia Code',Consolas,monospace;background:var(--bg);color:var(--text);font-size:12.5px;line-height:1.55;-webkit-font-smoothing:antialiased;min-height:100vh}
body::after{content:'';position:fixed;inset:0;background:radial-gradient(ellipse 60% 40% at 50% -10%,rgba(92,200,255,.025),transparent);pointer-events:none;z-index:0}

/* ── topbar ── */
.topbar{position:sticky;top:0;z-index:20;background:var(--panel);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;align-items:center;gap:10px}
.topbar .logo{font-weight:800;font-size:13px;color:var(--accent);letter-spacing:1.5px}
.topbar .sep{color:var(--dim);margin:0 4px}
.topbar .mode{font-size:10px;color:var(--dim);border:1px solid var(--border);padding:2px 8px;letter-spacing:1px}
.topbar .stat-inline{display:flex;gap:16px;margin-left:auto;font-size:10px}
.topbar .stat-inline span{color:var(--dim)}
.topbar .stat-inline b{color:var(--accent);font-weight:800}
@media(max-width:700px){.topbar .stat-inline{display:none}}

/* ── main grid ── */
.main{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:20px 16px;display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}
@media(max-width:800px){.main{grid-template-columns:1fr;padding:14px 10px;gap:12px}}

/* ── panel ── */
.panel{border:1px solid var(--border);background:var(--panel)}
.panel-head{padding:10px 14px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;background:rgba(92,200,255,.015)}
.panel-head h2{font-size:11px;color:var(--accent);letter-spacing:1.5px;font-weight:800}
.panel-head .hint{color:var(--dim);font-size:10px}
.panel-body{padding:14px}

/* ── form elements ── */
select,textarea,input[type=text]{width:100%;background:var(--bg);border:1px solid var(--border);padding:8px 10px;color:var(--text);font-family:inherit;font-size:12px;line-height:1.5}
select{padding:7px 10px;cursor:pointer;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%233d5268'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center;padding-right:28px}
select:focus,textarea:focus,input:focus{outline:none;border-color:var(--accent)}
textarea{min-height:80px;resize:vertical}
textarea::placeholder{color:var(--dim)}
.row{display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap}

/* ── buttons ── */
.btn{display:inline-flex;align-items:center;gap:5px;padding:7px 14px;border:1px solid var(--border);background:transparent;color:var(--text);cursor:pointer;font-family:inherit;font-size:11px;font-weight:700;letter-spacing:.8px;white-space:nowrap;transition:all .12s}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn-accent{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.btn-accent:hover{background:transparent;color:var(--accent)}
.btn-sm{padding:5px 10px;font-size:10px}

/* ── table ── */
.tbl{width:100%;border-collapse:collapse;font-size:11px}
.tbl thead{border-bottom:2px solid var(--border)}
.tbl th{padding:7px 6px;text-align:left;color:var(--dim);font-weight:700;font-size:9.5px;letter-spacing:.8px;white-space:nowrap}
.tbl td{padding:7px 6px;border-bottom:1px solid rgba(21,32,46,.6);word-break:break-all}
.tbl tr:hover td{background:var(--row-hover)}
@media(max-width:500px){.tbl th,.tbl td{font-size:10px;padding:6px 4px}}

/* ── badge ── */
.badge{display:inline-flex;align-items:center;gap:3px;padding:1px 7px;font-size:9.5px;font-weight:700;letter-spacing:.4px;white-space:nowrap}
.badge::before{content:'';width:4px;height:4px}
.badge-on{color:var(--green);border:1px solid rgba(63,185,80,.35)}.badge-on::before{background:var(--green)}
.badge-off{color:var(--red);border:1px solid rgba(248,81,73,.3)}.badge-off::before{background:var(--red)}
.badge-idle{color:var(--dim);border:1px solid var(--border)}.badge-idle::before{background:var(--dim)}

/* ── response area ── */
#response{background:var(--bg);border:1px solid var(--border);border-top:none;padding:12px;min-height:120px;max-height:400px;overflow-y:auto;font-size:12px;line-height:1.6;white-space:pre-wrap}
#response:empty::after{content:'等待发送…';color:var(--dim)}
.response-status{display:flex;justify-content:space-between;padding:6px 10px;font-size:10px;border-bottom:1px solid var(--border)}
.response-status .ok{color:var(--green)}.response-status .err{color:var(--red)}

/* ── toast ── */
.toast{position:fixed;top:20px;right:20px;z-index:99;padding:10px 18px;font-size:11px;font-weight:700;letter-spacing:.5px;animation:slide .25s;border:1px solid}
.toast-ok{background:rgba(63,185,80,.1);color:var(--green);border-color:rgba(63,185,80,.25)}
.toast-err{background:rgba(248,81,73,.1);color:var(--red);border-color:rgba(248,81,73,.25)}
@keyframes slide{from{transform:translateY(-8px);opacity:0}to{transform:translateY(0);opacity:1}}

/* ── misc ── */
.empty{color:var(--dim);padding:20px 0;text-align:center;font-size:11px}
.hidden{display:none}
.spacer{flex:1}
.ellipsis{max-width:160px;overflow:hidden;text-overflow:ellipsis;display:block}
.help{font-size:10px;color:var(--dim);margin-bottom:8px;opacity:.7}
.imp{padding:12px;margin-top:12px}
@media(max-width:500px){.hide-mobile{display:none}}
</style>
</head>
<body>
<div class="topbar">
  <span class="logo">▸ DS2API</span>
  <span class="mode">浏览器模式</span>
  <div class="stat-inline" id="topStats">
    <span>账号 <b>—</b></span>
    <span>活跃 <b>—</b></span>
    <span>可用 <b>—</b></span>
    <span>在线 <b>—</b></span>
    <span>排队 <b>—</b></span>
  </div>
</div>

<div class="main">

  <!-- ═══ 左栏：接口测试 ═══ -->
  <div class="panel" style="grid-row:span 1">
    <div class="panel-head">
      <h2>接口测试</h2>
      <span class="hint">/v1/chat/completions</span>
    </div>
    <div class="panel-body">
      <div class="row">
        <select id="model" style="flex:1">
          <option value="deepseek-flash">deepseek-flash</option>
          <option value="deepseek-pro">deepseek-pro</option>
        </select>
        <label style="font-size:11px;color:var(--dim);display:flex;align-items:center;gap:4px;white-space:nowrap">
          <input type="checkbox" id="stream" checked> 流式
        </label>
      </div>
      <textarea id="prompt" placeholder="输入消息…">你好，用一句话介绍你自己</textarea>
      <div class="row" style="margin-top:10px;margin-bottom:0">
        <button class="btn btn-accent" onclick="sendMsg()" id="sendBtn">▸ 发送</button>
        <button class="btn btn-sm" onclick="sendMsg()" id="sendBtn2" style="display:none">▸ 发送</button>
        <span id="reqStatus" style="font-size:10px;color:var(--dim)"></span>
        <div class="spacer"></div>
        <button class="btn btn-sm" onclick="document.getElementById('response').textContent=''">清空</button>
      </div>
      <div style="margin-top:12px;border:1px solid var(--border);border-bottom:none">
        <div class="response-status">
          <span id="respLabel">响应</span>
          <span id="respTime"></span>
        </div>
        <div id="response"></div>
      </div>
    </div>
  </div>

  <!-- ═══ 右栏：账号管理 ═══ -->
  <div class="panel">
    <div class="panel-head">
      <h2>账号管理</h2>
      <button class="btn btn-sm" onclick="loadAccounts()">刷新</button>
    </div>
    <div class="panel-body" style="padding-bottom:8px">
      <table class="tbl">
        <thead><tr><th>邮箱</th><th class="hide-mobile">备注</th><th>登录</th><th>状态</th><th>禁言</th><th class="hide-mobile">错误</th></tr></thead>
        <tbody id="tbl"><tr><td colspan="6" class="empty">加载中…</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="panel">
    <div class="panel-head">
      <h2>导入账号</h2>
    </div>
    <div class="panel-body">
      <div class="help">格式：邮箱:密码 ，每行一个</div>
      <textarea id="inp" placeholder="user@gmail.com:password&#10;user2@gmail.com:password" style="min-height:70px"></textarea>
      <div class="row" style="margin-top:10px;margin-bottom:0">
        <button class="btn btn-accent" onclick="doImport()">▸ 导入</button>
        <span id="msg" style="font-size:10px;color:var(--dim)"></span>
      </div>
    </div>
  </div>

</div>

<script>
const H=location.origin
const KEY='sbgptwcnmsbopenaiwdnmdcnmsbchat'

function toast(m,ok){
  const e=document.createElement('div')
  e.className='toast toast-'+(ok?'ok':'err')
  e.textContent=m
  document.body.appendChild(e)
  setTimeout(()=>e.remove(),2500)
}

async function api(p,o={}){
  const hd={};if(o.json)hd['Content-Type']='application/json'
  Object.assign(hd,o.headers||{})
  const r=await fetch(H+p,{headers:hd,method:o.method||'GET',body:o.body})
  if(!r.ok){const t=await r.text();throw new Error(t||r.status)}
  return r.json()
}

/* ── 接口测试 ── */
async function sendMsg(){
  const model=document.getElementById('model').value
  const prompt=document.getElementById('prompt').value.trim()
  const stream=document.getElementById('stream').checked
  const resp=document.getElementById('response')
  const status=document.getElementById('reqStatus')
  const timeEl=document.getElementById('respTime')
  const btn=document.getElementById('sendBtn')

  if(!prompt)return toast('请输入消息',0)
  btn.disabled=true;btn.textContent='…'
  resp.textContent='';timeEl.textContent=''

  const t0=Date.now()
  try{
    const r=await fetch(H+'/v1/chat/completions',{
      method:'POST',
      headers:{'Content-Type':'application/json','Authorization':'Bearer '+KEY},
      body:JSON.stringify({model,messages:[{role:'user',content:prompt}],stream})
    })

    if(stream){
      const reader=r.body.getReader(),dec=new TextDecoder()
      let full=''
      while(1){
        const{done,value}=await reader.read()
        if(done)break
        for(const line of dec.decode(value,{stream:true}).split('\\n')){
          if(!line.startsWith('data: '))continue
          const d=line.slice(6).trim()
          if(d==='[DONE]')continue
          try{const j=JSON.parse(d);const c=j.choices?.[0]?.delta?.content;if(c){full+=c;resp.textContent=full}}
          catch(e){}
        }
      }
      timeEl.textContent=((Date.now()-t0)/1000).toFixed(1)+'s'
      status.textContent='流式完成';status.className='ok'
    }else{
      const d=await r.json()
      resp.textContent=d.choices?.[0]?.message?.content||JSON.stringify(d,null,2)
      timeEl.textContent=((Date.now()-t0)/1000).toFixed(1)+'s'
      status.textContent=r.status+' OK';status.className='ok'
    }
  }catch(e){
    resp.textContent='错误: '+e.message
    status.textContent='失败';status.className='err'
  }
  btn.disabled=false;btn.textContent='▸ 发送'
}

/* ── 统计 & 账号 ── */
async function loadStats(){
  try{
    const s=await api('/readyz')
    document.getElementById('topStats').innerHTML=
      `<span>账号 <b>${s.accounts.total}</b></span>
       <span>活跃 <b>${s.accounts.in_use}</b></span>
       <span>可用 <b>${s.accounts.available}</b></span>
       <span>在线 <b>${s.accounts.logged_in}</b></span>
       <span>禁言 <b style="color:var(--red)">${s.accounts.muted||0}</b></span>
       <span>排队 <b>${s.accounts.queue_size}</b></span>`
  }catch(e){}
}
async function loadAccounts(){
  try{
    const d=await api('/admin/accounts',{headers:{'admin-key':'admin'}})
    let r=''
    for(const a of d.accounts){
      r+=`<tr>
        <td><span class="ellipsis">${a.email}</span></td>
        <td class="hide-mobile">${a.name||'—'}</td>
        <td><span class="badge ${a.logged_in?'badge-on':'badge-off'}">${a.logged_in?'在线':'离线'}</span></td>
        <td><span class="badge ${a.in_use?'badge-on':'badge-idle'}">${a.in_use?'使用中':'空闲'}</span></td>
        <td>${a.is_muted?`<span class="badge badge-off" title="${a.muted_until||'已禁言'}">禁言</span>`:'<span class="badge badge-idle">正常</span>'}</td>
        <td class="hide-mobile">${a.error_count>0?'<span class="badge badge-off">'+a.error_count+'</span>':'—'}</td>
      </tr>`
    }
    document.getElementById('tbl').innerHTML=r||'<tr><td colspan="6" class="empty">暂无账号</td></tr>'
  }catch(e){
    document.getElementById('tbl').innerHTML='<tr><td colspan="6" style="color:var(--red)">'+e.message+'</td></tr>'
  }
}
async function loadAll(){await loadStats();await loadAccounts()}

async function doImport(){
  const v=document.getElementById('inp').value.trim()
  if(!v)return toast('请输入账号',0)
  const accts=[]
  for(const l of v.split('\\n')){
    const t=l.trim();if(!t)continue
    const p=t.split(':',3)
    if(p.length>=2)accts.push({email:p[0].trim(),password:p[1],name:p[2]||''})
  }
  if(!accts.length)return toast('格式错误',0)
  try{
    const d=await api('/admin/accounts/import',{
      method:'POST',json:true,
      body:JSON.stringify({accounts:accts}),
      headers:{'admin-key':'admin'}
    })
    document.getElementById('inp').value=''
    document.getElementById('msg').textContent='已导入 '+d.imported+' 个'
    toast('成功导入 '+d.imported+' 个',1)
    loadAll()
  }catch(e){toast(e.message,0)}
}

// 回车发送
document.getElementById('prompt').addEventListener('keydown',e=>{
  if(e.ctrlKey&&e.key==='Enter')sendMsg()
})

loadAll()
setInterval(loadAll,12000)
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
