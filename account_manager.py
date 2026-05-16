import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

from deepseek_api import DeepSeekAPI


@dataclass
class Account:
    email: str
    password: str
    name: str = ""
    proxy: Optional[str] = None
    api: Optional[DeepSeekAPI] = field(default=None, repr=False)
    in_use_count: int = 0
    max_concurrent: int = 3
    error_count: int = 0
    logged_in: bool = False


class AccountManager:
    def __init__(self, max_concurrent_per_account: int = 3):
        self.accounts: Dict[str, Account] = {}
        self.queue: deque = deque()
        self.max_concurrent_per_account = max_concurrent_per_account
        self._lock = asyncio.Lock()

    def add_account(self, email: str, password: str, name: str = "", proxy: Optional[str] = None):
        self.accounts[email] = Account(
            email=email,
            password=password,
            name=name,
            proxy=proxy,
            max_concurrent=self.max_concurrent_per_account,
        )

    async def acquire(self) -> Account:
        async with self._lock:
            for account in self.accounts.values():
                if account.in_use_count < account.max_concurrent and account.error_count < 3:
                    account.in_use_count += 1
                    return account

        return await self._wait_for_account()

    async def _wait_for_account(self) -> Account:
        event = asyncio.Event()
        async with self._lock:
            self.queue.append(event)

        await event.wait()

        async with self._lock:
            for account in self.accounts.values():
                if account.in_use_count < account.max_concurrent and account.error_count < 3:
                    account.in_use_count += 1
                    return account

        raise RuntimeError("No account available")

    async def release(self, account: Account):
        async with self._lock:
            account.in_use_count = max(0, account.in_use_count - 1)
            if self.queue:
                event = self.queue.popleft()
                event.set()

    async def mark_error(self, account: Account):
        async with self._lock:
            account.error_count += 1
            account.in_use_count = max(0, account.in_use_count - 1)
            if self.queue:
                event = self.queue.popleft()
                event.set()

    async def get_api(self, account: Account) -> DeepSeekAPI:
        try:
            if account.api is None:
                account.api = DeepSeekAPI(
                    email=account.email,
                    password=account.password,
                    proxy=account.proxy,
                )
                await account.api.login()
                account.logged_in = True
            return account.api
        except Exception as e:
            print(f"Error creating API: {e}")
            await self.close_api(account)
            raise

    async def get_api_with_retry(self, account: Account) -> DeepSeekAPI:
        try:
            return await self.get_api(account)
        except Exception:
            await self.close_api(account)
            return await self.get_api(account)

    async def close_api(self, account: Account):
        if account.api:
            try:
                await account.api.close()
            except:
                pass
            account.api = None
            account.logged_in = False

    def get_stats(self) -> Dict:
        total = len(self.accounts)
        in_use = sum(a.in_use_count for a in self.accounts.values())
        available = sum(1 for a in self.accounts.values() if a.in_use_count < a.max_concurrent and a.error_count < 3)
        logged_in = sum(1 for a in self.accounts.values() if a.logged_in)
        return {
            "total": total,
            "in_use": in_use,
            "available": available,
            "logged_in": logged_in,
            "queue_size": len(self.queue),
            "max_concurrent_per_account": self.max_concurrent_per_account,
        }
