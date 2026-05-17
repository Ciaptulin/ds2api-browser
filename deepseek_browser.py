import asyncio
import logging
import random
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

from cloakbrowser import launch_persistent_context_async

logger = logging.getLogger(__name__)


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

    def _mask_email(self) -> str:
        """Generate a masked version of the email for skip_phrases filtering."""
        parts = self.email.split("@")
        if len(parts) == 2:
            local = parts[0]
            domain = parts[1]
            if len(local) > 4:
                masked = local[:4] + "*" * (len(local) - 4)
            else:
                masked = local[0] + "*" * (len(local) - 1)
            return f"{masked}@{domain}"
        return self.email

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
        # Wait for page ready instead of fixed sleep
        try:
            await self.page.wait_for_selector('textarea', timeout=15000)
        except Exception:
            await asyncio.sleep(2)

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

        # Check if account is muted after login
        if self._logged_in:
            await self._check_mute()

    async def _check_mute(self):
        """Check if account is muted and extract mute expiry."""
        try:
            muted, until = await self.page.evaluate("""() => {
                const text = document.body.innerText || '';
                // Match: 禁言至 YYYY年M月D日 HH:MM or 禁言至 YYYY-MM-DD HH:MM
                const match = text.match(/禁言至\\s*(\\d{4}[-年]\\d{1,2}[-月]\\d{1,2}[日]?\\s*\\d{1,2}:\\d{2})/);
                if (match) return [true, match[1]];
                if (text.includes('禁言')) return [true, ''];
                return [false, ''];
            }""")
            self._is_muted = muted
            self._muted_until = until
            if muted:
                logger.warning("[mute] %s is muted until %s", self.email, until)
        except Exception:
            self._is_muted = False
            self._muted_until = ""

    def is_muted(self) -> bool:
        return getattr(self, '_is_muted', False)

    def muted_until(self) -> str:
        return getattr(self, '_muted_until', "")

    async def _auto_login(self):
        logger.info("Logging in as %s...", self.email)

        try:
            email_input = self.page.locator('input[placeholder*="邮箱"], input[placeholder*="手机"], input[placeholder*="Email"], input[placeholder*="email"], input[type="text"]').first
            await email_input.wait_for(state="visible", timeout=10000)
            await email_input.fill(self.email)
            await asyncio.sleep(0.5)
        except Exception as e:
            # Take screenshot to debug
            try:
                await self.page.screenshot(path=f"/tmp/login_fail_{self.email.replace('@','_at_')}.png")
                logger.error("Screenshot saved to /tmp/login_fail_%s.png", self.email.replace('@', '_at_'))
            except Exception:
                pass
            logger.error("Email input error: %s", e)
            raise

        try:
            password_input = self.page.locator('input[type="password"]').first
            await password_input.wait_for(state="visible", timeout=5000)
            await password_input.fill(self.password)
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error("Password input error: %s", e)
            raise

        try:
            login_button = self.page.locator('button:has-text("登录")').first
            await login_button.click()
            await asyncio.sleep(3)
        except Exception as e:
            logger.error("Login button error: %s", e)
            raise

        try:
            await self.page.wait_for_selector('textarea', timeout=30000)
            self._logged_in = True
            self._ready = True
            logger.info("Login successful!")
        except Exception:
            raise Exception("Login failed")

    async def _human_delay(self, min_ms: int = 50, max_ms: int = 150):
        """Minimal delay for speed — just enough to avoid race conditions."""
        delay = random.uniform(min_ms, max_ms) / 1000
        await asyncio.sleep(delay)

    def _get_skip_phrases(self) -> list:
        """Build skip phrases list, dynamically including masked email."""
        phrases = [
            '深度思考', '智能搜索', '快速模式', '专家模式',
            '内容由 AI 生成', '开启新对话', '暂无历史对话', '今天',
        ]
        masked = self._mask_email()
        phrases.append(masked)
        return phrases

    async def new_chat(self):
        """Start a new chat by clicking the new-chat button instead of full page reload."""
        try:
            # Try clicking the "new chat" button first (much faster than goto)
            new_chat_btn = self.page.locator(
                'a:has-text("开启新对话"), button:has-text("开启新对话"), '
                'a:has-text("新对话"), button:has-text("新对话"), '
                '[class*="new-chat"], [class*="newChat"]'
            ).first
            if await new_chat_btn.count() > 0:
                await new_chat_btn.click()
                await self.page.wait_for_selector('textarea', timeout=10000)
                return

            # Fallback: full page reload
            await self.page.goto(self.DEEPSEEK_URL, timeout=30000)
            await self.page.wait_for_selector('textarea', timeout=15000)
        except Exception as e:
            logger.error("New chat error: %s", e)
            raise

    async def delete_chat(self):
        try:
            # Find the sidebar and active conversation
            chat_list = self.page.locator(
                'nav, aside, [class*="sidebar"], [class*="Sidebar"], div:has-text("开启新对话")'
            )
            chat_list_count = await chat_list.count()
            if chat_list_count == 0:
                logger.debug("[delete_chat] no sidebar")
                return

            active_item = chat_list.first.locator(
                '[class*="active"], [class*="selected"], [class*="current"]'
            ).first
            active_count = await active_item.count()
            if active_count == 0:
                # No active item yet (first chat), skip deletion
                logger.debug("[delete_chat] no active item, skipping")
                return

            # Get bounding box and click near right edge where "..." should be
            box = await active_item.bounding_box()
            if not box:
                logger.debug("[delete_chat] no bbox")
                return

            # Instead of position-based click, find the "..." element in DOM
            click_result = await self.page.evaluate("""() => {
                // Find the active/highlighted conversation item
                const active = document.querySelector('[class*="active"], [class*="selected"]');
                if (!active) return 'no-active';
                
                // Walk down to find a clickable child that looks like "..."
                // The "..." is often a button or div with no text (SVG only)
                const walk = (node, depth) => {
                    if (depth > 10) return null;
                    for (const child of node.children || []) {
                        const tag = child.tagName;
                        const cls = (child.className || '').toString();
                        // Look for small icon-like elements
                        if ((tag === 'BUTTON' || tag === 'svg' || cls.includes('icon') || cls.includes('more') || cls.includes('menu') || cls.includes('action')) && 
                            child.offsetWidth < 40 && child.offsetWidth > 0) {
                            return child;
                        }
                        const found = walk(child, depth + 1);
                        if (found) return found;
                    }
                    return null;
                };
                
                const icon = walk(active, 0);
                if (icon) {
                    icon.click();
                    return 'clicked:' + icon.tagName + ':' + (icon.className || '').substring(0, 40);
                }
                
                // Fallback: find any button/svg in active item
                const btn = active.querySelector('button, [role="button"]');
                if (btn) {
                    btn.click();
                    return 'fallback:' + btn.tagName;
                }
                return 'no-icon';
            }""")
            logger.debug("[delete_chat] icon click: %s", click_result)
            await asyncio.sleep(0.5)

            # Search for "删除" or "Delete" anywhere on page
            delete_btn = self.page.locator(
                ':has-text("删除"), :has-text("Delete")'
            ).first
            delete_count = await delete_btn.count()
            
            if delete_count == 0:
                logger.debug("[delete_chat] no delete option found")
                return

            await delete_btn.click()
            await asyncio.sleep(0.5)

            # Confirm
            confirm_btn = self.page.locator(
                'button:has-text("确认"), button:has-text("删除"), '
                'button:has-text("Confirm"), button:has-text("Delete")'
            ).last
            if await confirm_btn.count() > 0:
                await confirm_btn.click()
                await asyncio.sleep(1)
                logger.debug("[delete_chat] done!")
            else:
                logger.debug("[delete_chat] no confirm btn")
                
        except Exception as e:
            logger.warning("[delete_chat] error: %s", e)

    async def switch_model(self, model: str):
        try:
            # 极速思考模式
            if 'fast' in model or 'lite' in model:
                fast_btn = self.page.locator(
                    'button:has-text("极速思考"), div:has-text("极速思考"), '
                    'button:has-text("快速模式"), div:has-text("快速模式")'
                ).first
                if await fast_btn.count() > 0:
                    await fast_btn.click()
                    await asyncio.sleep(0.5)
            # 深度思考 (DeepThink R1)
            elif 'reasoner' in model or 'thinking' in model or 'pro' in model:
                thinking_btn = self.page.locator(
                    'button:has-text("深度思考"), div:has-text("深度思考"), '
                    'button:has-text("DeepThink"), button:has-text("R1")'
                ).first
                if await thinking_btn.count() > 0:
                    await thinking_btn.click()
                    await asyncio.sleep(0.5)

            # 专家模式 (独立开关)
            if 'expert' in model:
                expert_btn = self.page.locator(
                    'button:has-text("专家模式"), div:has-text("专家模式")'
                ).first
                if await expert_btn.count() > 0:
                    await expert_btn.click()
                    await asyncio.sleep(0.5)

            if 'search' in model:
                search_btn = self.page.locator('button:has-text("智能搜索"), div:has-text("智能搜索")').first
                if await search_btn.count() > 0:
                    await search_btn.click()
                    await asyncio.sleep(0.5)
        except Exception:
            pass

    async def send_message(self, prompt: str, timeout: int = 120, model: str = "deepseek-chat") -> dict:
        """Send message and return {'content': str, 'reasoning_content': str}."""
        try:
            await self.new_chat()
            await self.switch_model(model)

            input_field = self.page.locator('textarea').first
            await input_field.wait_for(state="visible", timeout=15000)

            await input_field.fill(prompt)
            await self._human_delay()
            await input_field.press('Enter')

            result = await self._wait_for_response(timeout, prompt)

            asyncio.create_task(self._safe_delete_chat())

            return result
        except Exception as e:
            logger.error("Send message error: %s", e)
            raise

    async def _safe_delete_chat(self):
        """Non-blocking delete chat wrapper."""
        try:
            await self.delete_chat()
        except Exception as e:
            logger.debug("[safe_delete] %s", e)

    # JavaScript to extract thinking and answer content from DOM
    _EXTRACT_JS = """() => {
        const result = {thinking: '', answer: '', done: false};

        // Find all assistant message containers (last one is current)
        const msgs = document.querySelectorAll(
            '[class*="assistant"], [class*="bot-"], [class*="message--"]:not([class*="user"])'
        );
        const lastMsg = msgs.length ? msgs[msgs.length - 1] : null;
        const scope = lastMsg || document.body;

        // Extract thinking/reasoning content
        const thinkEls = scope.querySelectorAll(
            '[class*="think"], [class*="Think"], [class*="reasoning"], details, [class*="collapse"]'
        );
        for (const el of thinkEls) {
            const t = el.innerText.trim();
            if (t && t.length > 5) {
                result.thinking = t.replace(/^.*深度思考[（(].*?[)）].*$/m, '').trim();
                break;
            }
        }

        // Extract answer content (markdown blocks outside thinking)
        const mdEls = scope.querySelectorAll(
            '[class*="markdown"], [class*="Markdown"], [class*="answer"], [class*="content"]'
        );
        for (const el of mdEls) {
            if (el.closest('[class*="think"], [class*="Think"], [class*="reasoning"], details')) continue;
            const t = el.innerText.trim();
            if (t && t.length > 2) {
                result.answer = t;
                break;
            }
        }

        // Fallback: parse body text if DOM selectors missed it
        const bodyText = scope.innerText || '';
        if (!result.answer || (!result.thinking && (bodyText.includes('深度思考') || bodyText.includes('思考过程')))) {
            const lines = bodyText.split('\\n').map(l => l.trim()).filter(Boolean);
            const skip = ['智能搜索', '快速模式', '专家模式', '极速思考',
                          '内容由 AI 生成', '开启新对话', '暂无历史对话'];
            
            let isThinking = false;
            let thinkLines = [];
            let ansLines = [];
            
            for (const l of lines) {
                if (skip.some(s => l === s)) continue;
                
                if (l === '深度思考' || l === '思考过程' || l.startsWith('深度思考...') || l.startsWith('极速思考...')) {
                    isThinking = true;
                    continue;
                }
                if (l.startsWith('已深度思考') || l.startsWith('深度思考（用时') || l.startsWith('已极速思考') || l.startsWith('极速思考（用时')) {
                    isThinking = false;
                    continue;
                }
                
                if (isThinking) {
                    thinkLines.push(l);
                } else {
                    ansLines.push(l);
                }
            }
            
            result.thinking = thinkLines.join('\\n');
            result.answer = ansLines.join('\\n');
        }

        // Check if response is complete
        const stopBtn = document.querySelector('[class*="stop"], button[aria-label*="stop"]');
        result.done = (!stopBtn || stopBtn.offsetParent === null);

        return result;
    }"""

    async def _wait_for_response(self, timeout: int, prompt: str = "") -> dict:
        """Wait for response and return {content, reasoning_content}."""
        deadline = time.time() + timeout
        await asyncio.sleep(0.8)

        last_answer = ""
        last_thinking = ""
        stable_count = 0

        while time.time() < deadline:
            try:
                result = await self.page.evaluate(self._EXTRACT_JS)
                # DEBUG LOG
                logger.debug("Extraction result: %s", str(result)[:200])
                
                answer = (result.get("answer") or "").strip()
                thinking = (result.get("thinking") or "").strip()

                if answer or thinking:
                    if answer != last_answer or thinking != last_thinking:
                        last_answer = answer
                        last_thinking = thinking
                        stable_count = 0
                    else:
                        stable_count += 1

                    if stable_count >= 3:
                        return {"content": last_answer, "reasoning_content": last_thinking}

            except Exception:
                pass

            await asyncio.sleep(0.5)

        if last_answer or last_thinking:
            return {"content": last_answer, "reasoning_content": last_thinking}

        raise TimeoutError("No response received")

    async def stream_message(self, prompt: str, timeout: int = 120, model: str = "deepseek-chat") -> AsyncGenerator[dict, None]:
        """Stream response, yielding dicts: {'type': 'thinking'|'content', 'chunk': str}."""
        try:
            await self.new_chat()
            await self.switch_model(model)

            input_field = self.page.locator('textarea').first
            await input_field.wait_for(state="visible", timeout=15000)

            await input_field.fill(prompt)
            await self._human_delay()
            await input_field.press('Enter')

            deadline = time.time() + timeout
            last_thinking = ""
            last_answer = ""
            stable_count = 0

            await asyncio.sleep(0.8)

            while time.time() < deadline:
                try:
                    result = await self.page.evaluate(self._EXTRACT_JS)
                    # DEBUG LOG
                    logger.debug("Stream extraction result: %s", str(result)[:200])

                    thinking = (result.get("thinking") or "").strip()
                    answer = (result.get("answer") or "").strip()

                    if thinking and thinking != last_thinking:
                        new_think = thinking[len(last_thinking):]
                        if new_think:
                            yield {"type": "thinking", "chunk": new_think}
                        last_thinking = thinking

                    if answer and answer != last_answer:
                        new_ans = answer[len(last_answer):]
                        if new_ans:
                            yield {"type": "content", "chunk": new_ans}
                        last_answer = answer
                        stable_count = 0
                    elif answer:
                        stable_count += 1

                    if stable_count >= 3:
                        break

                except Exception:
                    pass

                await asyncio.sleep(0.3)

            try:
                await self.delete_chat()
            except Exception as e:
                logger.warning("[stream_message] delete_chat cleanup error: %s", e)

        except Exception as e:
            logger.error("Stream message error: %s", e)
            raise

    async def close(self):
        if self.context:
            await self.context.close()
