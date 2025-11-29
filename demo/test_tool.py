import asyncio
import os

import tyro
from loguru import logger
from pydantic import Field

from mira.args import OpenAIArgs
from mira.openrouter import OpenRouterLLM
from mira.types import HumanMessage, LLMTool


class AddTools(LLMTool):
    """calculate the sum of two numbers"""

    x: float = Field(..., description="the first number")
    y: float = Field(..., description="the second number")

    def __call__(self):
        return self.x + self.y


class MultiplyTools(LLMTool):
    """calculate the product of two numbers"""

    x: float = Field(..., description="the first number")
    y: float = Field(..., description="the second number")

    def __call__(self):
        return self.x * self.y


async def main(args: OpenAIArgs):
    llm = OpenRouterLLM(args=args)
    m = [HumanMessage(content="calculate 34 + 2356 and calculate 467 * 12, then calculate the sum of the two results")]
    dm = await llm.invoke(messages=m, tools=[AddTools, MultiplyTools])
    dm = await llm.invoke(messages=m + dm[0], tools=[AddTools, MultiplyTools])
    for ddm in dm:
        logger.info(ddm)


if __name__ == "__main__":
    args = tyro.cli(OpenAIArgs)
    args.api_key = os.getenv("ONEAPI_API_KEY")
    args.base_url = os.getenv("ONEAPI_BASE_URL")
    logger.info(args)

    asyncio.run(main(args))
