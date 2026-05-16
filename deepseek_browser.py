import asyncio
import random
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
from cloakbrowser import launch_persistent_context_async


class DeepSeekBrowser:
    DEEPSEEK_URL = "https://chat.deepseek.com"
    LOGIN_URL = "https://chat.deepseek.com/api/v0/users/login"
    CREATE_SESSION_URL = "https://chat.deepseek.com/api/v0/chat_session/create"
    COMPLETION_URL = "https://chat.deepseek.com/api/v0/chat/completion"

    def __init__(
        self,
        email: str,
        password: str,
        profile_dir: str = "./profiles",
        headless: bool = True,
        humanize: bool = True,
        proxy: Optional[str] = None,
    ):
        self.email = email
        self.password = password
        self.profile_dir = Path(profile_dir) / email.replace("@", "_at_").replace("+", "_plus_")
        self.headless = headless
        self.humanize = humanize
        self.proxy = proxy
        self.context = None
        self.page = None
        self._logged_in = False
        self._ready = False
        self._token = None
        self._session_id = None

    async def start(self):
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        # 先用 API 登录获取 token
        await self._login_via_api()

        self.context = await launch_persistent_context_async(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            humanize=self.humanize,
            proxy=self.proxy,
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
        )

        self.page = await self.context.new_page()
        
        # 设置 token cookie
        if self._token:
            await self.context.add_cookies([{
                "name": "token",
                "value": self._token,
                "domain": ".deepseek.com",
                "path": "/",
            }])

        await self.page.goto(self.DEEPSEEK_URL, timeout=60000)
        await asyncio.sleep(3)

        # 检查是否登录成功
        if '/sign_in' in self.page.url:
            # 如果 cookie 方式失败，尝试通过 JS 注入 token
            await self.page.evaluate(f"localStorage.setItem('token', '{self._token}')")
            await self.page.reload()
            await asyncio.sleep(3)

        if '/sign_in' not in self.page.url:
            self._logged_in = True
            self._ready = True
        else:
            raise Exception("Login failed - still on sign_in page")

    async def _login_via_api(self):
        """通过 DeepSeek API 登录获取 token"""
        async with httpx.AsyncClient() as client:
            device_id = str(uuid.uuid4())
            payload = {
                "email": self.email,
                "password": self.password,
                "device_id": device_id,
                "os": "android",
            }
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "DeepSeek-Android/2.0",
            }
            
            resp = await client.post(self.LOGIN_URL, json=payload, headers=headers, timeout=30)
            data = resp.json()
            
            code = data.get("code", -1)
            if code != 0:
                msg = data.get("msg", "Unknown error")
                raise Exception(f"API login failed: {msg}")
            
            biz_data = data.get("data", {})
            biz_code = biz_data.get("biz_code", -1)
            if biz_code != 0:
                biz_msg = biz_data.get("biz_msg", "Unknown error")
                raise Exception(f"API login failed: {biz_msg}")
            
            user = biz_data.get("biz_data", {}).get("user", {})
            self._token = user.get("token", "")
            
            if not self._token:
                raise Exception("No token received from API")

    async def _human_delay(self, min_ms: int = 300, max_ms: int = 1500):
        delay = random.uniform(min_ms, max_ms) / 1000
        await asyncio.sleep(delay)

    async def new_chat(self):
        try:
            await self.page.goto(self.DEEPSEEK_URL, timeout=30000)
            await asyncio.sleep(2)
            await self.page.wait_for_selector('textarea', timeout=15000)
        except Exception as e:
            print(f"New chat error: {e}")
            raise

    async def delete_chat(self):
        try:
            more_btn = self.page.locator('button:has-text("更多"), .ds-icon-button:has-text("...")').first
            if await more_btn.count() > 0:
                await more_btn.click()
                await asyncio.sleep(0.5)

                delete_btn = self.page.locator('button:has-text("删除"), div:has-text("删除对话")').first
                if await delete_btn.count() > 0:
                    await delete_btn.click()
                    await asyncio.sleep(0.5)

                    confirm_btn = self.page.locator('button:has-text("确认"), button:has-text("删除")').last
                    if await confirm_btn.count() > 0:
                        await confirm_btn.click()
                        await asyncio.sleep(1)
        except Exception:
            pass

    async def switch_model(self, model: str):
        try:
            if 'reasoner' in model or 'thinking' in model:
                thinking_btn = self.page.locator('button:has-text("深度思考"), div:has-text("深度思考")').first
                if await thinking_btn.count() > 0:
                    await thinking_btn.click()
                    await asyncio.sleep(0.5)

            if 'search' in model:
                search_btn = self.page.locator('button:has-text("智能搜索"), div:has-text("智能搜索")').first
                if await search_btn.count() > 0:
                    await search_btn.click()
                    await asyncio.sleep(0.5)
        except Exception:
            pass

    async def send_message(self, prompt: str, timeout: int = 120, model: str = "deepseek-chat") -> str:
        try:
            await self.new_chat()
            await self.switch_model(model)

            input_field = self.page.locator('textarea').first
            await input_field.wait_for(state="visible", timeout=15000)

            await self._human_delay(500, 2000)

            await input_field.clear()
            await input_field.type(prompt, delay=random.randint(30, 80))

            await self._human_delay(200, 800)

            await input_field.press('Enter')

            response = await self._wait_for_response(timeout, prompt)

            await self.delete_chat()

            return response
        except Exception as e:
            print(f"Send message error: {e}")
            raise

    async def _wait_for_response(self, timeout: int, prompt: str = "") -> str:
        deadline = time.time() + timeout

        await asyncio.sleep(3)

        last_text = ""
        stable_count = 0

        skip_phrases = ['深度思考', '智能搜索', '快速模式', '专家模式', '内容由 AI 生成', '开启新对话', '暂无历史对话']

        while time.time() < deadline:
            try:
                try:
                    response_elements = await self.page.query_selector_all('.ds-markdown--block')
                    if response_elements:
                        last_response = response_elements[-1]
                        current_text = await last_response.inner_text()
                        current_text = current_text.strip()
                    else:
                        main_content = await self.page.query_selector('main, .chat-container, [class*="chat"]')
                        if main_content:
                            current_text = await main_content.inner_text()
                        else:
                            current_text = await self.page.inner_text('body')
                except Exception:
                    current_text = await self.page.inner_text('body')

                lines = current_text.split('\n')
                filtered_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if any(phrase in line for phrase in skip_phrases):
                        continue
                    filtered_lines.append(line)

                if filtered_lines:
                    current_text = '\n'.join(filtered_lines)

                    if current_text != last_text:
                        last_text = current_text
                        stable_count = 0
                    else:
                        stable_count += 1

                    if stable_count >= 3:
                        return current_text.strip()

            except Exception:
                pass

            await asyncio.sleep(0.5)

        if last_text:
            return last_text.strip()

        raise TimeoutError("No response received")

    async def stream_message(self, prompt: str, timeout: int = 120, model: str = "deepseek-chat") -> AsyncGenerator[str, None]:
        try:
            await self.new_chat()
            await self.switch_model(model)

            input_field = self.page.locator('textarea').first
            await input_field.wait_for(state="visible", timeout=15000)

            await self._human_delay(500, 2000)

            await input_field.clear()
            await input_field.type(prompt, delay=random.randint(30, 80))

            await self._human_delay(200, 800)

            await input_field.press('Enter')

            deadline = time.time() + timeout
            last_text = ""
            stable_count = 0

            skip_phrases = ['深度思考', '智能搜索', '快速模式', '专家模式', '内容由 AI 生成', '开启新对话', '暂无历史对话']

            await asyncio.sleep(3)

            while time.time() < deadline:
                try:
                    try:
                        response_elements = await self.page.query_selector_all('.ds-markdown--block')
                        if response_elements:
                            last_response = response_elements[-1]
                            current_text = await last_response.inner_text()
                            current_text = current_text.strip()
                        else:
                            main_content = await self.page.query_selector('main, .chat-container, [class*="chat"]')
                            if main_content:
                                current_text = await main_content.inner_text()
                            else:
                                current_text = await self.page.inner_text('body')
                    except Exception:
                        current_text = await self.page.inner_text('body')

                    lines = current_text.split('\n')
                    filtered_lines = []
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        if any(phrase in line for phrase in skip_phrases):
                            continue
                        filtered_lines.append(line)

                    if filtered_lines:
                        current_text = '\n'.join(filtered_lines)

                        if current_text != last_text:
                            new_chunk = current_text[len(last_text):]
                            if new_chunk:
                                yield new_chunk
                            last_text = current_text
                            stable_count = 0
                        else:
                            stable_count += 1

                        if stable_count >= 3:
                            return

                except Exception:
                    pass

                await asyncio.sleep(0.3)
        except Exception as e:
            print(f"Stream message error: {e}")
            raise

    async def close(self):
        if self.context:
            await self.context.close()
