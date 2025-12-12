import asyncio
import json
import queue
import threading
from concurrent.futures import Future
from typing import Any, List, Optional, Union

import httpx
import json_repair
from httpx_sse import aconnect_sse
from loguru import logger

from mira.args import OpenAIArgs, OpenRouterArgs
from mira.types import AIMessage, HumanMessage, NameSpace, SystemMessage, Task, ToolMessage

# class ToolThreadPool:
#     """
#     Elegant Thread Pool:
#     - Supports add() to add class tasks (non-blocking)
#     - Supports synchronous get_result()
#     - Supports asynchronous await get_result_async()
#     - Supports class initialization + __call__()
#     """

#     def __init__(self, num_workers: int = 4):
#         self.task_queue = queue.Queue()
#         self.workers = []
#         self._shutdown = False
#         self.tid = 0

#         for _ in range(num_workers):
#             t = threading.Thread(target=self._worker_loop, daemon=True)
#             t.start()
#             self.workers.append(t)

#     def _worker_loop(self):
#         while not self._shutdown:
#             try:
#                 task: Task = self.task_queue.get(timeout=0.1)
#             except queue.Empty:
#                 continue

#             if task.future.cancelled():
#                 continue

#             try:
#                 instance = task.cls(**task.args)

#                 result = instance.invoke(task.tid)

#                 task.future.set_result([result])
#             except Exception as e:
#                 task.future.set_exception(e)

#             self.task_queue.task_done()

#     def add(self, cls, args, tid) -> Future:
#         future = Future()
#         task = Task(cls=cls, args=args, tid=tid, future=future)
#         self.task_queue.put(task)
#         return future

#     async def get_result(self, future: Future):
#         loop = asyncio.get_running_loop()
#         return await loop.run_in_executor(None, future.result)

#     def shutdown(self):
#         self._shutdown = True
#         for t in self.workers:
#             t.join()


class OpenRouterClient:
    def __init__(self, api_key, base_url, provider: str):
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider

    def response_alignment(self, data: NameSpace):
        if self.provider == "doubao":
            for choice in data.choices:
                choice.delta.reasoning = choice.delta.reasoning_content
                choice.delta.reasoning_content = ""

        return data

    @logger.catch
    async def chat_completion_create(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[dict] = {"type": "text"},
        args: Optional[OpenRouterArgs | OpenAIArgs] = None,
        **kwargs,
    ):
        args = args.set_params(**kwargs)

        base_url = self.base_url + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://www.ai.com",
            "X-Title": f"{self.provider} Chat Completion",
        }
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "stream": args.stream,
        }

        payload.update(**args.to_dict())

        if self.provider in ["openai"]:
            payload["structured_outputs"] = response_format
        elif self.provider in ["doubao"]:
            payload["response_format"] = response_format
            payload["thinking"] = {"type": "enabled" if args.reasoning_effort else "disabled"}
            if "ep-" in args.model:
                payload["model"] = payload["model"].split("/")[-1]
        else:
            payload["response_format"] = response_format

        if args and args.verbose:
            logger.info(json.dumps(payload, indent=2, ensure_ascii=False))

        timeout = httpx.Timeout(3600.0)
        async with httpx.AsyncClient(timeout=timeout) as cli:
            if not args.stream:
                response = await cli.post(base_url, headers=headers, json=payload)
                data = response.json()
                if args.verbose:
                    logger.info(data)
                yield NameSpace(data)
            else:
                async with aconnect_sse(cli, "POST", base_url, headers=headers, json=payload) as s:
                    if s.response.status_code != 200:
                        logger.error(await s.response.aread())

                    async for sse in s.aiter_sse():
                        try:
                            if not sse.data:
                                continue
                            if sse.data == "[DONE]":
                                break
                            data = json.loads(sse.data)
                            if args.verbose:
                                logger.info(data)
                            data = NameSpace(data)
                            data = self.response_alignment(data)
                            yield data
                        except Exception as e:
                            logger.error(e)
                            continue


class OpenRouterLLM:
    # use asyncio
    def __init__(self, args: OpenRouterArgs = None, num_workers=4, **kwargs):
        self.args = args.set_params(**kwargs)
        self.semaphore = asyncio.Semaphore(num_workers)
        self.messages = []
        self.set_model()

    def set_model(self, **kwargs):
        if not self.args.model:
            raise ValueError("model is required")

        provider, model = self.args.model.split("/")

        api_key = self.args.api_key
        base_url = self.args.base_url
        self.model = provider + "/" + model
        self.provider = provider
        self.client = OpenRouterClient(api_key, base_url, provider)

        return self

    async def add_execute_callback(
        self,
        tools: List[Any],
        queue: List,
        **kwargs,
    ) -> asyncio.Task:
        if not tools:
            return None

        funcs = dict([(t.__name__, t) for t in tools])

        async def ready_to_fire(tool_calls):
            if self.args.stream:
                callids = dict([(t.index, t.id) for t in tool_calls if t.id])
                names = dict([(t.index, t.function.name) for t in tool_calls if t.function.name])
                arguments = [(t.index, t.function.arguments) for t in tool_calls]
            else:
                callids = dict([(i, t.id) for i, t in enumerate(tool_calls)])
                names = dict([(i, t.function.name) for i, t in enumerate(tool_calls)])
                arguments = [(i, t.function.arguments) for i, t in enumerate(tool_calls)]

            tasks = []
            for index in range(len(callids)):
                name = names[index]
                tool_call_id = callids[index]
                args = map(lambda x: x[1], filter(lambda x: x[0] == index and x[1], arguments))
                args = json_repair.loads("".join(args))
                task = funcs[name](**args).invoke(tool_call_id)
                tasks.append(task)

            return await asyncio.gather(*tasks)

        # return [[a, b], [a, b],..]
        gather = asyncio.create_task(ready_to_fire(queue))

        return gather

    async def generate(
        self,
        messages: List[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]] = [],
        tools: Optional[List[Any]] = [],
        response_format: Optional[Any] = None,
        **kwargs,
    ):
        if response_format:
            if isinstance(response_format, str):
                response_format = {"type": "regex", "pattern": response_format}
            else:
                response_format = response_format.schema()

        # define an async queue
        rollouts = {}

        async with self.semaphore:
            # 请求各路大模型 api，然后返回 json 格式的 response
            completion = self.client.chat_completion_create(
                model=self.model,
                messages=[m.dict() for m in messages],
                tools=[tool.schema() for tool in tools] or None,
                response_format=response_format,
                tool_choice="auto" if tools else None,
                args=self.args.set_params(**kwargs),
            )

            async for chunk in completion:
                if not chunk.choices:
                    continue

                # parse json format response
                for choice in chunk.choices:
                    index = choice.index

                    # In stream mode, `delta` provides incremental updates;
                    # in non-stream mode, `message` contains the complete content.
                    if choice.delta:
                        delta = choice.delta
                    else:
                        delta = choice.message

                    # If rollouts does not contain the index, initialize a rollout
                    if index not in rollouts:
                        rollouts[index] = NameSpace(
                            {
                                "queue": [],
                                "callback": None,
                                "messages": [],
                                "content": "",
                                "reasoning": "",
                                "logprobs": [],
                                "token_ids": [],
                            }
                        )

                    if delta.logprobs:
                        rollouts[index].logprobs.extend(delta.logprobs)

                    if delta.token_ids:
                        rollouts[index].token_ids.extend(delta.token_ids)

                    if delta.reasoning:
                        rollouts[index].reasoning += delta.reasoning
                        yield NameSpace({"index": index, "reasoning": delta.reasoning})

                    queue_i = rollouts[index].queue
                    if delta.tool_calls:
                        queue_i.extend(delta.tool_calls)

                    if choice.finish_reason == "tool_calls":
                        if queue_i:
                            callback = await self.add_execute_callback(tools, queue_i)
                            rollouts[index].callback = callback

                    if delta.content:
                        rollouts[index].content += delta.content
                        yield NameSpace({"index": index, "content": delta.content})

            for _, rollout in rollouts.items():
                callback = rollout.callback
                if callback:
                    tools_results = await callback
                    for tool_call, tool_return in tools_results:
                        tool_call.content = rollout.reasoning or rollout.content
                        tool_call.logprobs = rollout.logprobs
                        tool_call.token_ids = rollout.token_ids
                        rollout.messages.append(tool_call)
                        rollout.messages.append(tool_return)
                else:
                    if rollout.content:
                        assistant = AIMessage(
                            content=rollout.content,
                            logprobs=rollout.logprobs,
                            reasoning=rollout.reasoning,
                        )
                        rollout.messages.append(assistant)

        self.messages = [r.messages for r in rollouts.values()]

    async def forward(
        self,
        messages: List[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]] = [],
        tools: Optional[List[Any]] = [],
        response_format: Optional[Any] = None,
        **kwargs,
    ):
        generator = self.generate(
            messages=messages,
            tools=tools,
            response_format=response_format,
            **kwargs,
        )
        async for chunk in generator:
            pass

        return self.messages
