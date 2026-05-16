#!/usr/bin/env python3
"""Quick test script for DS2API Browser with multiple accounts."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# 多账号配置，用分号分隔
# 格式: email:password:name:proxy
os.environ["DS2API_ACCOUNTS"] = "huanxiangnb+dja@gmail.com:m1234567:账号1;huanxiangnb+321fffww@gmail.com:m1234567:账号2"
os.environ["DS2API_KEYS"] = "sk-test123456"
os.environ["DS2API_ADMIN_KEY"] = "admin"
os.environ["DS2API_PORT"] = "5002"
os.environ["DS2API_HEADLESS"] = "true"

from main import main

if __name__ == "__main__":
    main()
