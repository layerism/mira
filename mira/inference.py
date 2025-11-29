import asyncio
import copy
import inspect
import time
import uuid
from abc import ABC
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, List, Literal, Optional

import json_repair
import numpy as np
import torch
from loguru import logger
from lxml.etree import XMLPullParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import AsyncLLMEngine, LLMEngine, SamplingParams
from vllm.outputs import CompletionOutput
from vllm.sampling_params import StructuredOutputsParams

from mira.args import HFArgs, VLLMArgs


@dataclass
class ExtCompletionOutput(CompletionOutput):
    tool_calls: Optional[List[dict]] = field(default_factory=lambda: [])
    token_level_entropy: Optional[List[float]] = field(default_factory=lambda: [])


class StreamToolParser:
    def __init__(self):
        self.parser = XMLPullParser(events=("start", "end"), recover=True)
        self.parser.feed("<stream_root>")  # make a fake root
        self.index = -1

    async def feed(self, delta: Any, mode: str = "stream"):
        self.parser.feed(delta.text)

        datas = []
        for event, elem in self.parser.read_events():
            if event == "end" and elem.tag == "tool_call":
                self.index += 1
                text = (elem.text or "").strip()
                data = json_repair.loads(text)
                data["arguments"] = str(data["arguments"])
                datas.append(
                    {
                        "function": data,
                        "id": f"call_{uuid.uuid4().hex}",
                        "index": self.index,
                        "type": "function",
                    }
                )

                if mode == "stream":
                    elem.clear()
                    if elem.getparent() is not None:
                        elem.getparent().remove(elem)

                    return datas

        return datas


class Metric:
    def __init__(self, *args, **kwargs) -> None:
        self.dts = defaultdict(list)
        self.tokens = defaultdict(list)
        self.step_start = time.time()

    def update(self, key, n_tokens) -> Any:
        t = time.time()

        if key not in self.dts:
            self.dts[key] = [0]

        if key not in self.tokens:
            self.tokens[key] = [0]

        self.tokens[key].append(n_tokens)
        self.dts[key].append(t - self.step_start)

    def info(self) -> Any:
        ttfts, tpss = [], []
        for key in self.dts:
            ts = self.dts[key]
            tokens = self.tokens[key]
            ttfts.append(ts[-1] * 1000)
            tpss.append((tokens[-1] - tokens[1]) / (ts[-1] - ts[1]))

        logger.info(f"engine.ttft: {np.mean(ttfts):0.2f} ms")
        logger.info(f"engine.tps: {np.mean(tpss):0.2f} tokens/s")


class Engine(ABC):
    def __init__(self, args: None | HFArgs | VLLMArgs, **kwargs) -> None:
        self.args = args.set_params(**kwargs)
        self.engine = None
        self.tokenizer = None
        self.image_tokenizer = None
        self.audio_tokenizer = None

    def get_sampling_params(self, **kwargs):
        pass

    def generate(self, prompts: List[str], **kwargs) -> List[Any]:
        pass

    def get_logprobas(self, prompts: List[str], gen_texts: List[str], **kwargs) -> List[torch.Tensor]:
        pass

    def get_hidden_states(self, prompts: List[str], **kwargs) -> List[torch.Tensor]:
        pass

    def get_attentions(self, prompts: List[str], **kwargs) -> List[torch.Tensor]:
        pass

    def do_tokenize(self, prompts: List[str] | str, **kwargs) -> List[Any]:
        prompt_ids = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            padding_side="left",
            max_length=self.args.max_model_len,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        ).to(self.engine.device)

        return prompt_ids

    def apply_chat_template(
        self,
        messages: List[Any],
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[str | dict] = None,
        reasoning_effort: Optional[str] = None,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> List[str]:
        # then convert to dict
        if not isinstance(messages[0], dict):
            messages = [m.dict() for m in messages]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=True if reasoning_effort else False,
        )

        return prompt


class HFEngine(Engine):
    def __init__(self, args: HFArgs, **kwargs) -> None:
        self.args = args.set_params(**kwargs)
        self.engine = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.args.model,
            torch_dtype=torch.float16,
            offload_folder=args.offload_folder,
            device_map=args.device,
            max_memory=args.max_memory,
            trust_remote_code=args.trust_remote_code,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            attn_implementation=args.attn_implementation,
        )
        self.engine.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        self.dummy_prompt = None

        assert self.tokenizer.pad_token_id != self.tokenizer.eos_token_id
        assert self.tokenizer.pad_token_id != self.tokenizer.bos_token_id

    def get_sampling_params(self, **kwargs):
        args = copy.deepcopy(self.args).set_params(**kwargs)
        sampling_params = {
            "max_length": args.max_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "early_stopping": False,
            "num_beams": args.num_beams,
            "do_sample": args.do_sample,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": args.repetition_penalty,
            "num_return_sequences": args.n,
            "output_scores": args.output_scores,
            "return_dict_in_generate": args.return_dict_in_generate,
        }

        logger.info(f"sampling_params: {sampling_params}")
        return sampling_params

    def generate(self, prompts: List[str], **kwargs) -> List[Any]:
        inputs = self.do_tokenize(prompts)

        sampling_params = self.get_sampling_params(**kwargs)
        n = sampling_params["num_return_sequences"]
        pad_token_id = self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.engine.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                use_cache=True,
                **sampling_params,
            )

        responses = []
        for i in range(len(prompts)):
            completions = []
            for k in range(n):
                l = inputs.input_ids.shape[1]
                gen_tokens = outputs[i * n + k][l:]
                gen_tokens = gen_tokens[gen_tokens != pad_token_id]

                text = self.tokenizer.decode(
                    gen_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=True,
                )

                completions.append(
                    CompletionOutput(
                        index=k,
                        text=text,
                        token_ids=gen_tokens.cpu().tolist(),
                        cumulative_logprob=None,
                        finish_reason="stop",
                        logprobs=None,
                    )
                )

            responses.append(completions)

        return responses

    def get_logprobas(
        self,
        prompts: List[str],
        gen_texts: List[str],
        mode: Literal["nxm", "1x1"] = "1x1",
        **kwargs,
    ) -> List[torch.Tensor]:
        if not gen_texts:
            return []

        cmpl_prompts = []

        if mode == "1x1":
            assert len(prompts) == len(gen_texts)
            for prompt, gen_text in zip(prompts, gen_texts):
                cmpl_prompts.append(prompt + gen_text)
        elif mode == "nxm":
            for prompt in prompts:
                for gen_text in gen_texts:
                    cmpl_prompts.append(prompt + gen_text)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        inputs = self.do_tokenize(cmpl_prompts)

        with torch.no_grad():
            outputs = self.engine(**inputs, output_logits=True, return_dict_in_generate=True)
            logprobas = outputs.logits.log_softmax(-1)  # [bs, len, vocab_size]
            logprobas = logprobas.gather(dim=2, index=inputs.input_ids.unsqueeze(2)).squeeze(2)

        gen_logprobas = []
        for i in range(len(prompts)):
            prompt_ids = self.do_tokenize(prompts[i])
            l = prompt_ids.input_ids.shape[-1]
            logp = logprobas[i, l:].mean(-1)
            gen_logprobas.append(logp)

        return torch.stack(gen_logprobas)

    def get_attentions(self, prompts: List[str], **kwargs) -> List[torch.Tensor]:
        inputs = self.do_tokenize(prompts)

        with torch.no_grad():
            outputs = self.engine(**inputs, output_attentions=True, use_cache=False)

        return [attn.cpu() for attn in outputs.attentions]

    def get_past_kv(self, prompts: List[str], **kwargs) -> List[torch.Tensor]:
        inputs = self.do_tokenize(prompts)

        with torch.no_grad():
            outputs = self.engine(
                **inputs,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=True,
            )
            past_kv = outputs.past_key_values

        return past_kv


# class PrefillEngine(Engine):

#     def __init__(self, args: ServerArgs, **kwargs) -> None:
#         self.args = args.set_params(**kwargs)
#         self.model = self.args.model
#         self.engine_args = self.get_engine_args(args)
#         self.engine = LLMEngine.from_engine_args(self.engine_args)
#         self.tokenizer = self.engine.tokenizer.tokenizer


class VLLMEngine(Engine):
    def __init__(self, args: VLLMArgs, **kwargs) -> None:
        self.args = args.set_params(**kwargs)
        self.model = self.args.model
        self.engine_args = self.get_engine_args(args)
        self.engine = LLMEngine.from_engine_args(self.engine_args)
        self.tokenizer = self.engine.tokenizer.tokenizer
        self.dummy_prompt = None

    def get_engine_args(self, args: VLLMArgs, enabled_async: bool = False):
        if enabled_async:
            from vllm import AsyncEngineArgs as EngineArgs
        else:
            from vllm import EngineArgs

        engine_keys = inspect.signature(EngineArgs).parameters.keys()
        engine_args = {k: v for k, v in args.to_dict().items() if k in engine_keys}
        engine_args = EngineArgs(**engine_args)

        logger.info(f"engine_args: {engine_args}")

        return engine_args

    def get_sampling_params(self, response_format: Optional[dict] = None, **kwargs):
        args = copy.deepcopy(self.args).set_params(**kwargs)
        sampling_keys = inspect.signature(SamplingParams).parameters.keys()
        sampling_args = {k: v for k, v in args.to_dict().items() if k in sampling_keys}
        sampling_params = SamplingParams(**sampling_args)

        if response_format:
            schema = response_format["json_schema"]["schema"]
            so = StructuredOutputsParams(json=schema)
            sampling_params.structured_outputs = so

        logger.info(f"sampling_params: {sampling_params}")

        return sampling_params

    def add_request(self, prompt: str, response_format: Optional[dict] = None, **kwargs) -> str:
        # lstrip bos_token because vllm will add it.
        if self.tokenizer.bos_token:
            prompt = prompt.lstrip(self.tokenizer.bos_token)

        sampling_params = self.get_sampling_params(response_format, **kwargs)

        request_id = "req_" + str(uuid.uuid4())
        self.engine.add_request(request_id, prompt, sampling_params)

        return request_id

    def generate(self, prompts: List[str], response_format: Optional[Any] = None, **kwargs) -> List[Any]:
        if isinstance(prompts, str):
            prompts = [prompts]

        request_id_list = [self.add_request(prompt, response_format, **kwargs) for prompt in prompts]

        n = kwargs.get("n", None) or self.args.n

        outputs, count = defaultdict(list), 0
        while self.engine.has_unfinished_requests():
            request_outputs = self.engine.step()

            for req_out in request_outputs:
                if req_out.request_id not in request_id_list:
                    continue
                for output in req_out.outputs:
                    if output.finish_reason:
                        outputs[req_out.request_id].append(output)
                        count += 1

                if count == len(prompts) * n:
                    break

        outputs = [outputs[req_id] for req_id in request_id_list]
        outputs = [sorted(output, key=lambda x: x.index) for output in outputs]

        if self.args.logprobs:
            for output in outputs:
                for token_output in output:
                    token_logprobs = [
                        logprobs[ids].logprob
                        for ids, logprobs in zip(
                            token_output.token_ids,
                            token_output.logprobs,
                        )
                    ]
                    token_output.logprobs = token_logprobs

        if len(prompts) == 1:
            outputs = outputs[0]

        return outputs


class AsyncVLLMEngine(VLLMEngine):
    def __init__(self, args: VLLMArgs, **kwargs):
        self.args = args.set_params(**kwargs)
        self.model = self.args.model
        self.engine_args = self.get_engine_args(args, enabled_async=True)
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        self.tokenizer = None
        self.q = {}

    async def add_request(self, prompt: str, response_format: Optional[dict] = None, **kwargs) -> str:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        sampling_params = self.get_sampling_params(response_format, **kwargs)

        # lstrip bos_token because vllm will add it.
        if self.tokenizer.bos_token:
            prompt = prompt.lstrip(self.tokenizer.bos_token)

        logger.info(f"prompt: {repr(prompt)}")

        if request_id not in self.q:
            self.q[request_id] = asyncio.Queue()

        async def generator():
            async for req_out in self.engine.generate(prompt, sampling_params, request_id):
                if req_out is not None:
                    await self.q[request_id].put(req_out)
            # ending flag
            await self.q[request_id].put(None)

        asyncio.create_task(generator())

        return request_id

    async def completion(self, request_id: str) -> Any:
        tc_parser = defaultdict(lambda: StreamToolParser())

        while True:
            req_out = await self.q[request_id].get()
            if req_out is None:
                break

            for output in req_out.outputs:
                if output.finish_reason:
                    if output.logprobs:
                        pairs = list(zip(output.token_ids, output.logprobs))
                        output.logprobs = [logprob[ids].logprob for ids, logprob in pairs]
                    else:
                        output.logprobs = []
                        output.cumulative_logprob = 0.0

                    output = ExtCompletionOutput(**copy.deepcopy(asdict(output)))

                    tool_calls = await tc_parser[output.index].feed(output, mode=None)
                    if tool_calls:
                        output.text = ""
                        output.tool_calls = tool_calls
                        output.finish_reason = "tool_calls"

                    yield output

    async def delta(self, request_id: str) -> Any:
        tc_parser = defaultdict(lambda: StreamToolParser())
        last_output = defaultdict(lambda: None)

        while True:
            req_out = await self.q[request_id].get()
            if req_out is None:
                break

            for output in req_out.outputs:
                delta = copy.deepcopy(output)
                logger.error(delta)

                prev_text = ""
                prev_token_ids = []
                if last_output[delta.index]:
                    prev_text = last_output[delta.index].text
                    prev_token_ids = last_output[delta.index].token_ids

                delta.text = delta.text[len(prev_text) :]
                delta.token_ids = delta.token_ids[len(prev_token_ids) :]

                if delta.logprobs:
                    delta.logprobs = delta.logprobs[len(prev_token_ids) :]
                    pairs = list(zip(delta.token_ids, delta.logprobs))
                    delta.logprobs = [logprob[ids].logprob for ids, logprob in pairs]
                    delta.text = "".join([logprob[ids].decoded_token for ids, logprob in pairs])
                else:
                    delta.logprobs = []
                    delta.cumulative_logprob = 0.0

                delta = ExtCompletionOutput(**asdict(delta))

                tool_calls = await tc_parser[delta.index].feed(delta)
                if tool_calls:
                    delta.tool_calls = tool_calls
                    delta.finish_reason = "tool_calls"

                yield delta

                last_output[delta.index] = copy.deepcopy(output)

    async def shutdown(self) -> None:
        await self.engine.shutdown()
