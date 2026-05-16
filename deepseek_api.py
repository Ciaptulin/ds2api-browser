import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from typing import AsyncGenerator, Optional, Union

import httpx

logger = logging.getLogger(__name__)


class DeepSeekAPI:
    LOGIN_URL = "https://chat.deepseek.com/api/v0/users/login"
    CREATE_SESSION_URL = "https://chat.deepseek.com/api/v0/chat_session/create"
    CREATE_POW_URL = "https://chat.deepseek.com/api/v0/chat/create_pow_challenge"
    COMPLETION_URL = "https://chat.deepseek.com/api/v0/chat/completion"
    DELETE_SESSION_URL = "https://chat.deepseek.com/api/v0/chat_session/delete"

    def __init__(self, email: str, password: str, proxy: Optional[str] = None):
        self.email = email
        self.password = password
        self.proxy = proxy
        self._token: Optional[str] = None
        self._device_id = str(uuid.uuid4())
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                proxy=self.proxy,
                timeout=60.0,
                follow_redirects=True,
            )
        return self._client

    async def login(self) -> str:
        client = await self._get_client()
        
        payload = {
            "email": self.email,
            "password": self.password,
            "device_id": self._device_id,
            "os": "android",
        }
        
        headers = self._base_headers()
        
        resp = await client.post(self.LOGIN_URL, json=payload, headers=headers, timeout=30)
        data = resp.json()
        
        code = data.get("code", -1)
        if code != 0:
            raise Exception(f"Login failed: {data.get('msg', 'Unknown')}")
        
        biz_data = data.get("data", {})
        biz_code = biz_data.get("biz_code", -1)
        if biz_code != 0:
            raise Exception(f"Login failed: {biz_data.get('biz_msg', 'Unknown')}")
        
        user = biz_data.get("biz_data", {}).get("user", {})
        self._token = user.get("token", "")
        
        if not self._token:
            raise Exception("No token received")
        
        return self._token

    async def create_session(self) -> str:
        if not self._token:
            await self.login()
        
        client = await self._get_client()
        headers = self._auth_headers()
        
        resp = await client.post(
            self.CREATE_SESSION_URL,
            json={"agent": "chat"},
            headers=headers,
            timeout=30,
        )
        data = resp.json()
        
        if data.get("code") != 0:
            raise Exception(f"Create session failed: {data.get('msg')}")
        
        biz_data = data.get("data", {}).get("biz_data", {})
        
        if "chat_session" in biz_data:
            session_id = biz_data["chat_session"].get("id", "")
        else:
            session_id = biz_data.get("id", "")
        
        if not session_id:
            raise Exception("No session ID received")
        
        return session_id

    async def get_pow(self, target_path: str = "/api/v0/chat/completion") -> dict:
        client = await self._get_client()
        headers = self._auth_headers()
        
        resp = await client.post(
            self.CREATE_POW_URL,
            json={"target_path": target_path},
            headers=headers,
            timeout=30,
        )
        data = resp.json()
        
        if data.get("code") != 0:
            raise Exception(f"Get PoW failed: {data.get('msg')}")
        
        return data.get("data", {}).get("biz_data", {}).get("challenge", {})

    def _compute_pow(self, challenge: dict) -> str:
        prefix = challenge.get("prefix", "")
        target = challenge.get("target", "")
        expire_at = challenge.get("expire_at", 0)
        
        nonce = 0
        while nonce < 10000000:
            test = f"{prefix}{nonce}"
            hash_val = hashlib.sha256(test.encode()).hexdigest()
            # Compare hash against target lexicographically — this handles
            # both all-zero targets and mixed targets like "000abc".
            if hash_val <= target:
                break
            nonce += 1
        
        return json.dumps({
            "prefix": prefix,
            "nonce": nonce,
            "expire_at": expire_at,
            "target": target,
        })

    async def send_message(
        self,
        prompt: str,
        model: str = "deepseek-chat",
        stream: bool = False,
        timeout: int = 120,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Send a message and return a response.

        Returns:
            str when stream=False, AsyncGenerator[str, None] when stream=True.
        """
        if not self._token:
            await self.login()
        
        session_id = await self.create_session()
        
        pow_challenge = await self.get_pow()
        pow_header = self._compute_pow(pow_challenge)
        
        client = await self._get_client()
        headers = self._auth_headers()
        headers["x-ds-pow-response"] = pow_header
        
        payload = {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": True,
            "search_enabled": "search" in model.lower(),
        }
        
        if stream:
            return self._stream_completion(client, headers, payload, timeout)
        else:
            return await self._sync_completion(client, headers, payload, timeout)

    async def _sync_completion(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        payload: dict,
        timeout: int,
    ) -> str:
        resp = await client.post(
            self.COMPLETION_URL,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        
        # 处理 SSE 响应
        full_text = ""
        for line in resp.text.split("\n"):
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                    msg = data.get("message", {})
                    content = msg.get("content", "")
                    if content:
                        full_text += content
                except json.JSONDecodeError:
                    continue
        
        return full_text

    async def _stream_completion(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        payload: dict,
        timeout: int,
    ) -> AsyncGenerator[str, None]:
        async with client.stream(
            "POST",
            self.COMPLETION_URL,
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        data = json.loads(data_str)
                        msg = data.get("message", {})
                        content = msg.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def delete_session(self, session_id: str):
        if not self._token:
            return
        
        client = await self._get_client()
        headers = self._auth_headers()
        
        await client.post(
            self.DELETE_SESSION_URL,
            json={"chat_session_id": session_id},
            headers=headers,
            timeout=30,
        )

    def _base_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "User-Agent": "DeepSeek/3.2.1 Android/35",
            "x-client-platform": "android",
            "x-client-version": "3.2.1",
            "x-client-locale": "zh_CN",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

    def _auth_headers(self) -> dict:
        headers = self._base_headers()
        if self._token:
            headers["authorization"] = f"Bearer {self._token}"
        return headers

    async def close(self):
        if self._client:
            await self._client.aclose()
