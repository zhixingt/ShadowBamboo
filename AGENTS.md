# AGENTS.md - Open WebUI Sub Agent Plugin

Open WebUI 插件，在独立子代理上下文中运行自主任务，保持主对话清洁。

## 技术栈

- **语言**: Python 3.10+ (单文件 `Sub Agent.py`)
- **框架**: FastAPI + Pydantic
- **运行环境**: Open WebUI v0.7.0+

## 开发命令

```powershell
# Lint 检查
ruff check "Sub Agent.py"

# 类型检查
mypy "Sub Agent.py" --ignore-missing-imports

# 格式化
black "Sub Agent.py"
```

测试需在 Open WebUI 环境中进行：放入 tools 目录 → 重启服务 → 调用工具。

## 代码风格

### 导入

```python
# 1. 标准库 (字母序)
import asyncio
import json
from typing import Any, Optional

# 2. 第三方库 (字母序)
from fastapi import Request
from pydantic import BaseModel, Field

# 3. Open WebUI 模块 (函数内延迟导入，避免非运行时错误)
from open_webui.utils.chat import generate_chat_completion
```

### 类型注解

```python
# 返回类型必须显式声明
def func(name: str, items: Optional[List[dict]] = None) -> str: ...

# Python 3.10+ 泛型语法
def process(tasks: list[dict]) -> dict: ...

# 所有工具方法必须是 async
async def run_sub_agent(self, ...) -> str: ...
```

### 命名约定

| 类型 | 约定 | 示例 |
|------|------|------|
| 类名 | PascalCase | `Tools`, `UserValves` |
| 函数/变量 | snake_case | `run_sub_agent`, `model_id` |
| 常量 | UPPER_SNAKE_CASE | `BUILTIN_TOOL_CATEGORIES` |
| 私有函数 | _leading_underscore | `_find_manifest_in_text` |
| Pydantic 字段 | UPPER_SNAKE_CASE | `MAX_ITERATIONS` |

### 错误处理

```python
# 记录异常堆栈
try:
    result = await tool_function(**params)
except Exception as e:
    log.exception(f"Error: {e}")
    return {"error": str(e)}

# JSON 错误响应
return json.dumps({"error": "message"}, ensure_ascii=False)
```

## Open WebUI 约定

### 特殊参数 (双下划线前缀)

```python
async def tool_method(
    self,
    __user__: dict = None,              # 用户信息
    __request__: Request = None,        # FastAPI 请求
    __model__: dict = None,             # 模型信息
    __metadata__: dict = None,          # 元数据
    __event_emitter__: Callable = None, # 事件发射器
    __messages__: Optional[list] = None,# 对话历史
) -> str:
```

### 返回格式

```python
# 工具结果必须是 JSON 字符串
return json.dumps({"result": data}, ensure_ascii=False)

# 状态更新
await __event_emitter__({
    "type": "status",
    "data": {"description": "Processing...", "done": False},
})
```

## 代码结构

```python
# ============================================================================
# Section Header
# ============================================================================

# 常量在文件顶部
CONSTANT_NAME = {"key": "value"}

# 辅助函数在类定义前
def helper_function(): ...

# Tools 是插件主类
class Tools:
    class Valves(BaseModel):
        """管理员配置"""
        SETTING: str = Field(default="", description="说明")

    class UserValves(BaseModel):
        """用户配置"""
        USER_SETTING: str = Field(default="")

    def __init__(self):
        self.valves = self.Valves()

    async def tool_method(self, ...): ...
```

## 依赖模块 (延迟导入)

- `open_webui.models.users` - 用户模型
- `open_webui.utils.tools` - 工具加载
- `open_webui.utils.chat` - 聊天完成
- `open_webui.utils.filter` - 过滤器
- `open_webui.tools.builtin` - 内置工具

这些模块仅在 Open WebUI 运行时可用。