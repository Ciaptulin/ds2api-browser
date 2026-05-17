import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

from deepseek_browser import DeepSeekBrowser

logger = logging.getLogger(__name__)


import time

@dataclass
class Account:
    email: str
    password: str
    name: str = ""
    proxy: Optional[str] = None
    browser: Optional[DeepSeekBrowser] = field(default=None, repr=False)
    in_use: bool = False
    error_count: int = 0
    logged_in: bool = False
    is_muted: bool = False
    muted_until: str = ""
    last_used: float = 0.0


class AccountManager:
    def __init__(self, max_inflight: int = 1, max_active_browsers: int = 5):
        self.accounts: Dict[str, Account] = {}
        self.queue: deque = deque()
        self.max_inflight = max_inflight
        self.max_active_browsers = max_active_browsers
        self._lock = asyncio.Lock()

    def add_account(self, email: str, password: str, name: str = "", proxy: Optional[str] = None):
        self.accounts[email] = Account(
            email=email,
            password=password,
            name=name,
            proxy=proxy,
        )

    async def acquire(self) -> Account:
        async with self._lock:
            for account in self.accounts.values():
                if not account.in_use and account.error_count < 3 and not account.is_muted:
                    account.in_use = True
                    account.last_used = time.time()
                    return account

        return await self._wait_for_account()

    async def _wait_for_account(self) -> Account:
        event = asyncio.Event()
        async with self._lock:
            self.queue.append(event)

        await event.wait()

        async with self._lock:
            for account in self.accounts.values():
                if not account.in_use and account.error_count < 3 and not account.is_muted:
                    account.in_use = True
                    account.last_used = time.time()
                    return account

        raise RuntimeError("No account available")

    async def release(self, account: Account):
        async with self._lock:
            account.in_use = False
            account.last_used = time.time()
            if self.queue:
                event = self.queue.popleft()
                event.set()

    async def mark_error(self, account: Account):
        async with self._lock:
            account.error_count += 1
            account.in_use = False
            if self.queue:
                event = self.queue.popleft()
                event.set()

    async def _enforce_browser_limit(self):
        active = [a for a in self.accounts.values() if a.browser is not None]
        if len(active) >= self.max_active_browsers:
            idle = [a for a in active if not a.in_use]
            if idle:
                idle.sort(key=lambda x: x.last_used)
                to_close = len(active) - self.max_active_browsers + 1
                for a in idle[:to_close]:
                    logger.info("Closing idle browser for %s to free memory", a.email)
                    await self.close_browser(a)

    async def get_or_create_browser(self, account: Account, headless: bool = True) -> DeepSeekBrowser:
        try:
            if account.browser is None:
                await self._enforce_browser_limit()
                account.browser = DeepSeekBrowser(
                    email=account.email,
                    password=account.password,
                    profile_dir="./profiles",
                    headless=headless,
                    humanize=True,
                    proxy=account.proxy,
                )
                await account.browser.start()
                account.logged_in = True
                # Check mute status
                account.is_muted = account.browser.is_muted()
                account.muted_until = account.browser.muted_until()
            return account.browser
        except Exception as e:
            logger.error("Error creating browser for %s: %s", account.email, e)
            await self.close_browser(account)
            raise

    async def get_or_create_browser_with_retry(self, account: Account, headless: bool = True) -> DeepSeekBrowser:
        try:
            return await self.get_or_create_browser(account, headless)
        except Exception:
            await self.close_browser(account)
            return await self.get_or_create_browser(account, headless)

    async def close_browser(self, account: Account):
        if account.browser:
            try:
                await account.browser.close()
            except Exception as e:
                logger.debug("Error closing browser for %s: %s", account.email, e)
            account.browser = None
            account.logged_in = False

    def get_stats(self) -> Dict:
        total = len(self.accounts)
        in_use = sum(1 for a in self.accounts.values() if a.in_use)
        available = sum(1 for a in self.accounts.values() if not a.in_use and a.error_count < 3 and not a.is_muted)
        logged_in = sum(1 for a in self.accounts.values() if a.logged_in)
        muted = sum(1 for a in self.accounts.values() if a.is_muted)
        accounts_list = [
            {
                "email": a.email,
                "name": a.name,
                "in_use": a.in_use,
                "logged_in": a.logged_in,
                "is_muted": a.is_muted,
                "muted_until": a.muted_until,
                "error_count": a.error_count,
            }
            for a in self.accounts.values()
        ]
        return {
            "total": total,
            "in_use": in_use,
            "available": available,
            "logged_in": logged_in,
            "muted": muted,
            "queue_size": len(self.queue),
            "accounts": accounts_list,
        }
