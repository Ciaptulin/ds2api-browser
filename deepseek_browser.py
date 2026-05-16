import asyncio
import random
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

from cloakbrowser import launch_persistent_context_async


class DeepSeekBrowser:
    DEEPSEEK_URL = "https://chat.deepseek.com"

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

    async def start(self):
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        self.context = await launch_persistent_context_async(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            humanize=self.humanize,
            proxy=self.proxy,
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
        )

        self.page = await self.context.new_page()
        await self.page.goto(self.DEEPSEEK_URL, timeout=60000)
        await asyncio.sleep(5)

        await self._check_login_state()

    async def _check_login_state(self):
        current_url = self.page.url

        if '/sign_in' in current_url:
            await self._auto_login()
        else:
            try:
                await self.page.wait_for_selector('textarea', timeout=10000)
                self._logged_in = True
                self._ready = True
            except Exception:
                await self._auto_login()

    async def _auto_login(self):
        print(f"Logging in as {self.email}...")

        try:
            email_input = self.page.locator('input[placeholder*="邮箱"], input[placeholder*="手机"], input[type="text"]').first
            await email_input.wait_for(state="visible", timeout=10000)
            await email_input.fill(self.email)
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Email input error: {e}")
            raise

        try:
            password_input = self.page.locator('input[type="password"]').first
            await password_input.wait_for(state="visible", timeout=5000)
            await password_input.fill(self.password)
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Password input error: {e}")
            raise

        try:
            login_button = self.page.locator('button:has-text("登录")').first
            await login_button.click()
            await asyncio.sleep(3)
        except Exception as e:
            print(f"Login button error: {e}")
            raise

        try:
            await self.page.wait_for_selector('textarea', timeout=30000)
            self._logged_in = True
            self._ready = True
            print("Login successful!")
        except Exception:
            raise Exception("Login failed")

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

        skip_phrases = ['深度思考', '智能搜索', '快速模式', '专家模式', '内容由 AI 生成', '开启新对话', '暂无历史对话', '今天', 'huan********dja@gmail.com']

        while time.time() < deadline:
            try:
                text = await self.page.inner_text('body')

                lines = text.split('\n')
                response_started = False
                response_text = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if line == '内容由 AI 生成，请仔细甄别':
                        break

                    if any(phrase in line for phrase in skip_phrases):
                        continue

                    if response_started:
                        response_text.append(line)

                    if prompt and prompt in line:
                        response_started = True

                if response_text:
                    current_text = '\n'.join(response_text)

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

            skip_phrases = ['深度思考', '智能搜索', '快速模式', '专家模式', '内容由 AI 生成', '开启新对话', '暂无历史对话', '今天', 'huan********dja@gmail.com']

            await asyncio.sleep(3)

            while time.time() < deadline:
                try:
                    text = await self.page.inner_text('body')

                    lines = text.split('\n')
                    response_started = False
                    response_text = []

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        if line == '内容由 AI 生成，请仔细甄别':
                            break

                        if any(phrase in line for phrase in skip_phrases):
                            continue

                        if response_started:
                            response_text.append(line)

                        if prompt and prompt in line:
                            response_started = True

                    if response_text:
                        current_text = '\n'.join(response_text)

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
