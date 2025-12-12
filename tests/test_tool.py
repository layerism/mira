import asyncio
import os

import tyro
from loguru import logger
from pydantic import Field

from mira import HumanMessage, LLMTool, OpenAIArgs, OpenRouterLLM

problem = "calculate 34 + 2356 and calculate 467 * 12, then calculate the sum of the two results"


class AddTool(LLMTool):
    """calculate the sum of two numbers"""

    x: float = Field(..., description="the first number")
    y: float = Field(..., description="the second number")

    def __call__(self):
        return self.x + self.y


class MultiplyTool(LLMTool):
    """calculate the product of two numbers"""

    x: float = Field(..., description="the first number")
    y: float = Field(..., description="the second number")

    def __call__(self):
        return self.x * self.y


async def main(args: OpenAIArgs):
    llm = OpenRouterLLM(args=args)
    m = [HumanMessage(content=problem)]

    for _ in range(1):
        dm = await llm.forward(messages=m, tools=[AddTool, MultiplyTool])
        m = m + dm[0]

    logger.info(m)


if __name__ == "__main__":
    args = tyro.cli(OpenAIArgs)
    args.api_key = os.getenv("ONEAPI_API_KEY")
    args.base_url = os.getenv("ONEAPI_BASE_URL")
    logger.info(args)

    asyncio.run(main(args))
