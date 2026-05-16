import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AccountConfig:
    email: str
    password: str
    name: str = ""
    proxy: Optional[str] = None


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5001
    admin_key: str = "admin"


@dataclass
class BrowserConfig:
    headless: bool = True
    humanize: bool = True
    timeout: int = 60000
    viewport_width: int = 1920
    viewport_height: int = 1080


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    accounts: List[AccountConfig] = field(default_factory=list)
    api_keys: List[str] = field(default_factory=lambda: ["sk-default"])

    @classmethod
    def from_env(cls) -> "Config":
        accounts = []
        
        account_str = os.getenv("DS2API_ACCOUNTS", "")
        if account_str:
            for acc in account_str.split(";"):
                parts = acc.split(":", 3)
                if len(parts) >= 2:
                    accounts.append(AccountConfig(
                        email=parts[0],
                        password=parts[1],
                        name=parts[2] if len(parts) > 2 else "",
                        proxy=parts[3] if len(parts) > 3 else None,
                    ))

        return cls(
            server=ServerConfig(
                host=os.getenv("DS2API_HOST", "0.0.0.0"),
                port=int(os.getenv("DS2API_PORT", "5001")),
                admin_key=os.getenv("DS2API_ADMIN_KEY", "admin"),
            ),
            browser=BrowserConfig(
                headless=os.getenv("DS2API_HEADLESS", "true").lower() == "true",
                humanize=os.getenv("DS2API_HUMANIZE", "true").lower() == "true",
                timeout=int(os.getenv("DS2API_TIMEOUT", "60000")),
            ),
            accounts=accounts,
            api_keys=os.getenv("DS2API_KEYS", "sk-default").split(","),
        )


def load_config() -> Config:
    return Config.from_env()
