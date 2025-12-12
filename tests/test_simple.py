import asyncio
import os

import tyro
from loguru import logger

from mira import HumanMessage, OpenAIArgs, OpenRouterLLM

prompt = """
解读如下人物：
https://p1.itc.cn/q_70/images03/20221218/ff4cc248739049718df3f172dd1299e0.jpeg
"""


async def main(args: OpenAIArgs):
    llm = OpenRouterLLM(args=args)

    m = [HumanMessage(content=prompt)]

    dm = await llm.forward(messages=m)
    for ddm in dm:
        logger.info(ddm)


if __name__ == "__main__":
    args = tyro.cli(OpenAIArgs)
    args.api_key = os.getenv("ONEAPI_API_KEY")
    args.base_url = os.getenv("ONEAPI_BASE_URL")
    logger.info(args)

    asyncio.run(main(args))
