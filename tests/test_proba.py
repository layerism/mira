import asyncio
import os
import re
from typing import List

import numpy as np
import tyro
from loguru import logger
from pydantic import Field

from mira import HumanMessage, LLMJson, OpenAIArgs, OpenRouterLLM

pattern = r"""最终计算结果：\d+,\d+$"""


class OutputFormat(LLMJson):
    answer: List[float] = Field(..., description="the answer of the question")
    reasoning: str = Field(..., description="the reasoning step of the answer")


async def main(args):
    llm = OpenRouterLLM(args=args)
    m = HumanMessage(content="计算 23 + 456.1 的结果")

    dm = await llm.forward(
        messages=[m],
        response_format=r"最终计算结果：\d+$",
    )
    logger.info(dm)


if __name__ == "__main__":
    args = tyro.cli(OpenAIArgs)
    args.api_key = os.getenv("ONEAPI_API_KEY")
    args.base_url = os.getenv("ONEAPI_BASE_URL")
    logger.info(args)

    asyncio.run(main(args))
