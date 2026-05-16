#!/usr/bin/env python3
"""DS2API Browser 启动入口。

账号和密钥从 .env 文件自动加载，格式见 .env.example。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from main import main

if __name__ == "__main__":
    main()
