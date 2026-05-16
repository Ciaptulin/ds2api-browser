# DS2API Browser

基于 CloakBrowser/Playwright 的 DeepSeek API 代理服务。

## 特性

- **浏览器自动化** - 使用真实浏览器访问 DeepSeek，无法被检测
- **OpenAI 兼容 API** - 支持 `/v1/chat/completions` 接口
- **流式响应** - 支持 SSE 流式输出
- **账号池管理** - 支持多账号轮询
- **人类行为模拟** - 模拟真实用户操作

## 安装

```bash
cd /home/huanx/code/ds2api-browser
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 安装 Playwright 浏览器（如果不用 CloakBrowser）
playwright install chromium
```

## 使用

### 方式 1: 环境变量配置

```bash
export DS2API_ACCOUNTS="email1@gmail.com:password1;email2@gmail.com:password2"
export DS2API_KEYS="sk-key1,sk-key2"
export DS2API_ADMIN_KEY="your-admin-key"
export DS2API_PORT="5001"
export DS2API_HEADLESS="true"
export DS2API_HUMANIZE="true"

python main.py
```

### 方式 2: 直接运行

```bash
python start.py
```

### 方式 3: 后台运行

```bash
nohup python main.py > /tmp/ds2api-browser.log 2>&1 &
```

## API 使用

### 聊天补全

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer sk-test123456" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 流式响应

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer sk-test123456" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5001/v1",
    api_key="sk-test123456"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## 健康检查

```bash
# 健康检查
curl http://localhost:5001/healthz

# 就绪检查
curl http://localhost:5001/readyz

# 管理统计
curl http://localhost:5001/admin/stats -H "admin-key: admin"
```

## 与原版 DS2API 的区别

| 特性 | 原版 DS2API | DS2API Browser |
|------|-------------|----------------|
| 实现方式 | HTTP 客户端 | 浏览器自动化 |
| 指纹检测 | 容易被检测 | 无法被检测 |
| 账号封禁 | 高风险 | 低风险 |
| 性能 | 快 | 较慢 |
| 资源占用 | 低 | 高 |

## 注意事项

1. **首次运行** - Playwright/CloakBrowser 会下载浏览器二进制文件（~200MB）
2. **资源占用** - 每个浏览器实例占用约 200-500MB 内存
3. **性能** - 浏览器自动化比直接 HTTP 慢，但更安全
4. **账号安全** - 建议使用小号测试，不要用主账号

## 文件结构

```
ds2api-browser/
├── main.py              # FastAPI 服务器
├── deepseek_browser.py  # 浏览器自动化核心
├── account_manager.py   # 账号池管理
├── config.py            # 配置管理
├── start.py             # 快速启动脚本
├── run.py               # 运行入口
├── requirements.txt     # 依赖列表
└── README.md            # 本文档
```
