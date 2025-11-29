# Mira: Model-Integrated Routing & Adaptation

Mira 是一种支持 vllm，transformers，第三方 api 接口的统一部署调用库，设计目的是为了更加自由地设计 agent，能够获取概率，注意力等复杂信息，用户可以自己使用第三方模型，或者自己部署模型，然后使用统一的接口调度设计 agent

## 🌟 Features

- 兼容 OpenAI 协议
- 更加方便地 Rollout 获取概率和 pass@K 样本
- 支持本地 vLLM、HF Transformers 模型
- 兼容 OpenAI、Claude、Gemini、OpenRouter、Qwen、Seed 等模型（第三方的通常拿不到概率和注意力）
- 支持采用 BaseModel 类实现自定义的函数 tool，提供 tool 运行的线程管理，方便用户设计重型、复杂、带状态的函数功能
- 支持用户自定义上下文工程处理，支持更加灵活的 agent 设计

## TODO-LIST
- [ ] HF Transformers 原生模型部署支持
- [ ] 能够输出 token level entropy 以及短序列的 attention
- [ ] 方便后续电路图生成，用于可解释
- [ ] 兼容 OpenAI harmony 协议

## 🚀 Installation

### Prerequisites
- Ubuntu >= 22.04 or Centos >= 7
- CUDA-compatible GPU (for local inference, better for cuda 12.4+)

### Using Conda (Recommended)

```bash
# Create a new environment
conda create -n mira python=3.11
conda activate mira

# Clone the repository
git clone https://github.com/yourusername/mira.git
cd mira

# Install dependencies
pip install --upgrade pip setuptools
pip install -e .
```

## ⚙️ Configuration

1.  **Environment Variables**: Copy the template to create your local config.

    ```bash
    cp .env.template .env
    ```

2.  **Edit `.env`**: Fill in your API keys and preferences.

    ```ini
    # Local Inference Settings
    CUDA_VISIBLE_DEVICES=0
    
    # API Keys (Fill as needed)
    OPENAI_API_KEY=sk-...
    OPENROUTER_API_KEY=sk-...
    HF_TOKEN=hf_...
    ```

## 📖 Usage

### vllm 服务启动

```python
python -m mira.oai_protocol --model Qwen/Qwen3-8B
```

### 2. OpenAI-Compatible Server

Start an API server that mimics OpenAI's interface, serving your local models or routing requests.

```bash
# Start the server (example command, adjust based on your entry point)
python -m mira.oai_protocol --model Qwen/Qwen3-8B --host 0.0.0.0 --port 8000
```

*Note: Check `mira/oai_protocol.py` for specific CLI arguments.*

### 3. Using Remote APIs

Mira can act as a client for various LLM providers.

```python
from mira.openrouter import OpenRouterClient

client = OpenRouterClient()
response = client.chat.completions.create(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Tell me a joke."}]
)
print(response.choices[0].message.content)
```

## 📂 Project Structure

```
mira/
├── demo/               # Example scripts and tests
├── mira/
│   ├── inference.py    # Core inference engines (vLLM, HF)
│   ├── oai_protocol.py # OpenAI API server implementation
│   ├── openrouter.py   # OpenRouter and remote API clients
│   ├── args.py         # Configuration and argument parsing
│   └── types.py        # Type definitions
├── .env.template       # Environment variable template
├── pyproject.toml      # Project dependencies and build config
└── README.md           # This file
```

## 🗑️ Uninstallation

```bash
pip uninstall mira
# If using Conda
conda env remove -n mira
```

## 📄 License

MIT License