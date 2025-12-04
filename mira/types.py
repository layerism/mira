import base64
import json
import re
import textwrap
from concurrent.futures import Future
from typing import Any, Dict, List, Literal, Optional

import httpx
from loguru import logger
from openai import pydantic_function_tool
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam as Content
from pydantic import BaseModel, Field
from urlextract import URLExtract


class Message(BaseModel):
    def dict(self):
        return self.model_dump(exclude_none=True)


class Function(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str | Dict[str, Any]] = None

    def __init__(self, name: str, arguments: Dict):
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        super().__init__(name=name, arguments=arguments)


class ToolCall(BaseModel):
    id: str = ""
    type: Literal["function"] = "function"
    function: Optional[Function] = None


class SystemMessage(Message):
    role: Literal["system"] = "system"
    content: Optional[str] = None


class AIMessage(Message):
    role: Literal["assistant"] = "assistant"
    name: Optional[str] = None
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    logprobs: Optional[List[float]] = Field(default=[], exclude=True)
    token_ids: Optional[List[int]] = Field(default=[], exclude=True)


class ToolMessage(Message):
    role: Literal["tool"] = "tool"
    tool_call_id: str = ""
    name: Optional[str] = None
    content: Optional[str | Dict[str, Any]] = None


class HumanMessage(Message):
    role: Literal["user"] = "user"
    name: Optional[str] = None
    content: Optional[str | List[Content]] = None
    detail: Optional[str] = "auto"

    def __init__(self, content: str):
        super().__init__(content=content)

        splits = self.mmsplit(self.content)

        if len(splits) > 1:
            self.content = []
            for part, dtype in splits:
                if dtype == "imurl":
                    content = self.encode_image(part)
                    content = f"data:image/jpeg;base64,{content}"
                    self.content.append({"type": "image_url", "image_url": {"url": content, "detail": self.detail}})
                else:
                    self.content.append({"type": "text", "text": part})

    def encode_image(self, image_path):
        if image_path.startswith("http") or image_path.startswith("ftp"):
            response = httpx.get(image_path)
            bytes = response.content
        else:
            with open(image_path, "rb") as image_file:
                bytes = image_file.read()

        return base64.b64encode(bytes).decode("utf-8")

    def mmsplit(self, text: str):
        e = URLExtract()
        urls = e.find_urls(text)
        img_ext = re.compile(r"\.(?:jpe?g|png|gif|bmp|webp|svg|ico|tiff?)", re.IGNORECASE)
        urls = [u for u in urls if img_ext.search(u)]

        if not urls:
            return [{"text": text}]  # no image url, return text

        splits = []
        for url in urls:
            idx = text.find(url)
            splits.append((text[:idx], "text"))
            splits.append((url, "imurl"))
            text = text[idx + len(url) :]
        splits.append((text, "text"))

        logger.info(splits)

        return splits


class LLMTool(BaseModel):
    # model_config = ConfigDict(extra='allow')

    @classmethod
    def schema(cls):
        description = "tool schema"
        if cls.__doc__:
            description = textwrap.dedent(cls.__doc__).strip()
        shema = pydantic_function_tool(cls, name=cls.__name__, description=description)
        shema["function"]["parameters"].pop("description")
        return shema

    @classmethod
    def pprint(cls):
        shema = cls.schema()
        print(json.dumps(shema, indent=2, ensure_ascii=False))

    async def invoke(self, tool_call_id: str = ""):
        result = self.__call__()

        arguments = self.model_dump(exclude_none=True)

        tool_call = AIMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id=tool_call_id,
                    function=Function(name=self.__class__.__name__, arguments=arguments),
                )
            ],
        )

        tool_return = ToolMessage(
            tool_call_id=tool_call_id,
            name=self.__class__.__name__,
            content=str(result),
        )

        outputs = [tool_call, tool_return]

        return outputs


class LLMJson(BaseModel):
    @classmethod
    def schema(cls):
        schema = cls.model_json_schema()
        schema.setdefault("additionalProperties", False)
        if "required" not in schema:
            schema["required"] = list(schema.get("properties", {}).keys())

        full_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": cls.__name__,
                "strict": True,
                "schema": schema,
            },
        }
        return full_schema

    @classmethod
    def pprint(cls):
        shema = cls.schema()
        print(json.dumps(shema, indent=2, ensure_ascii=False))


class NameSpace:
    """
    支持链式访问的命名空间类
    - 支持 .a.b.c 链式访问
    - 不存在的属性返回 None
    - 支持从 dict/JSON 初始化
    """

    def __init__(self, data: dict | None = None, **kwargs):
        if data:
            kwargs.update(data)

        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = NameSpace(value)
            elif isinstance(value, list):
                value = [NameSpace(item) if isinstance(item, dict) else item for item in value]
            setattr(self, key, value)

    def __getattr__(self, name):
        """不存在的属性返回 None"""
        return None

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items()}
        return f"NameSpace({attrs})"

    def __getitem__(self, key):
        """支持字典式访问"""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """支持字典式赋值"""
        setattr(self, key, value)

    def to_dict(self):
        """转换回字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, NameSpace):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, NameSpace) else item for item in value]
            else:
                result[key] = value
        return result

    def get(self, key, default=None):
        """类似字典的 get 方法"""
        return getattr(self, key, default)


class Task:
    """
    Each task consists of:
    - cls: The class to be initialized and called
    - args/kwargs: Arguments for initializing cls
    - future: Used to store the execution result, for use by get_result()
    """

    def __init__(self, cls, tid, args, future: Future):
        self.cls = cls
        self.tid = tid
        self.args = args
        self.future = future
