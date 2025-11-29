# Mira 安装和使用指南

## 安装

### 开发模式安装（推荐用于开发）

```bash
# 在项目根目录下
pip install -e .
```

### 正式安装

```bash
pip install .
```

### 安装可选依赖

```bash
# 安装监控功能
pip install -e ".[monitoring]"

# 安装所有可选依赖
pip install -e ".[all]"
```

## 使用命令行工具

安装完成后，你可以直接使用 `mira-server` 命令来启动 OAI Protocol 服务器：

```bash
# 基本用法
mira-server --model <model_path> --host 0.0.0.0 --port 8000

# 查看所有可用参数
mira-server --help
```

### 常用参数示例

```bash
# 启动服务器，指定模型和端口
mira-server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192

# 使用多 GPU
mira-server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --port 8000
```

## Python API 使用

### 导入类型

```python
from mira.types import HumanMessage, AIMessage, SystemMessage, LLMTool, LLMJson
from mira.args import OpenAIArgs, VLLMArgs
from mira.openrouter import OpenRouterLLM
from mira.inference import VLLM
```

### 使用示例

```python
import asyncio
from mira.types import HumanMessage, LLMTool
from mira.args import OpenAIArgs
from mira.openrouter import OpenRouterLLM
from pydantic import Field

# 定义工具
class AddTool(LLMTool):
    """计算两个数字的和"""
    x: float = Field(..., description="第一个数字")
    y: float = Field(..., description="第二个数字")
    
    def __call__(self):
        return self.x + self.y

# 使用 LLM
async def main():
    args = OpenAIArgs(
        api_key="your-api-key",
        base_url="https://api.openai.com/v1"
    )
    llm = OpenRouterLLM(args=args)
    
    messages = [HumanMessage(content="计算 1 + 2")]
    response = await llm.invoke(messages=messages, tools=[AddTool])
    print(response)

asyncio.run(main())
```

## 验证安装

### 验证包导入

```bash
python -c "from mira.types import HumanMessage; print('✓ 成功导入 HumanMessage')"
```

### 验证命令行工具

```bash
which mira-server
mira-server --help
```

## 卸载

```bash
pip uninstall mira
```

## 常见问题

### 1. 无法导入 `mira.types`

确保已经正确安装了包：
```bash
pip install -e .
```

### 2. `mira-server` 命令找不到

重新安装包：
```bash
pip uninstall mira
pip install -e .
```

### 3. 依赖冲突

建议使用虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

pip install -e .
```
