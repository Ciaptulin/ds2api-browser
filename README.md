---
title: DS2API Browser
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# DS2API Browser

基于 CloakBrowser 的 DeepSeek API 代理服务。

## 为什么选择 CloakBrowser？

CloakBrowser 是专为反检测设计的浏览器，通过 30/30 机器人检测测试。使用 CloakBrowser 可以：

- **完全隐藏自动化痕迹** - 无法被 DeepSeek 检测
- **避免账号封禁** - 模拟真实用户行为
- **通过 AWS WAF 验证** - 自动处理 CloudFront Token

## 特性

- **浏览器自动化** - 使用 CloakBrowser 访问 DeepSeek，无法被检测
- **OpenAI 兼容 API** - 支持 `/v1/chat/completions` 接口
- **Claude/Gemini/Ollama 兼容** - 多协议支持
- **流式响应** - 支持 SSE 流式输出
- **账号池管理** - 支持多账号轮询
- **Web 管理界面** - 在线导入账号

## 安装

```bash
# 克隆仓库
# git clone https://github.com/Ciaptulin/ds2api-browser.git
cd ds2api-browser

# 创建虚拟环境
python -m venv venv
# source venv/bin/activate
source venv/Scripts/activate # Win平台用这个

# 安装依赖
pip install -r requirements.txt
```

## 使用

### 方式 1: 直接运行

```bash
python run.py
```

### 方式 3: 后台运行

```bash
nohup python run.py > /tmp/ds2api-browser.log 2>&1 &
```

## Web 管理界面

访问 `http://localhost:5001/` 可以：

- 测试 API 请求
- 导入账号（支持批量）
- 查看账号状态

## API 使用

### 聊天补全

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer sk-test123456" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 流式响应

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer sk-test123456" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
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
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## 支持的模型

| 模型 ID | 说明 |
|---------|------|
| deepseek-v4-flash | 标准版（全局默认开启深度思考 R1） |
| deepseek-v4-pro | 专家版（全局默认开启深度思考 R1） |

## 健康检查

```bash
# 健康检查
curl http://localhost:5001/healthz

# 就绪检查
curl http://localhost:5001/readyz

# 管理统计
curl http://localhost:5001/admin/stats -H "admin-key: admin"
```

## 注意事项

1. **必须使用 CloakBrowser** - 其他浏览器（如原生 Playwright）会被检测
2. **资源占用** - 每个浏览器实例占用约 200-500MB 内存
3. **性能** - 浏览器自动化比直接 HTTP 慢，但更安全
4. **账号安全** - 建议使用小号测试，不要用主账号

## 文件结构

```
ds2api-browser/
├── main.py              # FastAPI 服务器（API路由与核心拦截层）
├── deepseek_browser.py  # Playwright/CloakBrowser 自动化爬虫核心
├── account_manager.py   # 账号池调度与封禁规避
├── config.py            # 配置管理
├── run.py               # 启动入口
├── .env.example         # 环境变量模板
├── requirements.txt     # 依赖列表
└── README.md            # 本文档
```
