import asyncio
import os

import tyro
from loguru import logger
from pydantic import Field

from mira import LLMJson, OpenAIArgs, OpenRouterLLM


class OutputFormat(LLMJson):
    answer: str = Field(..., description="the answer of the question")
    reasoning: str = Field(..., description="the reasoning step of the answer")


async def main(args):
    llm = OpenRouterLLM(args=args)
    m = HumanMessage(content="计算 34 + 2356 以及 467 * 12 的结果。")
    dm = await llm.forward(messages=[m], response_format=OutputFormat)
    logger.info(dm)


if __name__ == "__main__":
    args = tyro.cli(OpenAIArgs)
    args.api_key = os.getenv("ONEAPI_API_KEY")
    args.base_url = os.getenv("ONEAPI_BASE_URL")
    logger.info(args)

    asyncio.run(main(args))
