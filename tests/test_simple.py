import asyncio
import os

import tyro
from loguru import logger

from mira.args import OpenAIArgs
from mira.openrouter import OpenRouterLLM
from mira.types import HumanMessage

# prompt = Template("""\
# Output Format: The last part of your final response should be in the following format, \
# you should write the solution of the problem inside the `Solution` tag, \
# and output the final result (number in string type) in json format.

# Problem: {{question}}
# <Solution>
# write your solution here
# </Solution>
# ```json
# {'final_answer': ...}
# ```
# """)


# problem = """
# Let $\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radius of the smallest sphere that can contain each of the rectangular boxes that are elements of $\mathcal{B}$. The value of $r^2$ can be written as $\frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$.
# """

# 描述如下图片：https://pic.616pic.com/photoone/00/02/58/618cf527354c35308.jpg!/fw/1120

prompt = """
解读如下人物：
https://p1.itc.cn/q_70/images03/20221218/ff4cc248739049718df3f172dd1299e0.jpeg
"""


async def main(args: OpenAIArgs):
    llm = OpenRouterLLM(args=args)

    m = [
        HumanMessage(content=prompt),
    ]

    dm = await llm.forward(messages=m)
    for ddm in dm:
        logger.info(ddm)


if __name__ == "__main__":
    args = tyro.cli(OpenAIArgs)
    args.api_key = os.getenv("ONEAPI_API_KEY")
    args.base_url = os.getenv("ONEAPI_BASE_URL")
    logger.info(args)

    asyncio.run(main(args))
