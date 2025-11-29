import asyncio
import os

import tyro
from jinja2 import Template
from loguru import logger

from mira.args import OpenAIArgs
from mira.openrouter import OpenRouterLLM
from mira.types import HumanMessage

prompt = Template("""\
Experience: {{experience}}\n\nOutput Format: The last part of your final response should be in the following format, \
you should write the solution of the problem after the `Final Answer` tag, \
and output the result (number,float,etc) after `@@@` tag.\n\nProblem: {{question}}\nFinal Answer: ... @@@ ...
""")


problem = """
Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?
"""


async def main(args: OpenAIArgs):
    llm = OpenRouterLLM(args=args)

    m = [
        HumanMessage(content=prompt.render(question=problem)),
    ]
    dm = await llm.invoke(messages=m)
    for ddm in dm:
        logger.info(ddm)


if __name__ == "__main__":
    args = tyro.cli(OpenAIArgs)
    args.api_key = os.getenv("ONEAPI_API_KEY")
    args.base_url = os.getenv("ONEAPI_BASE_URL")
    logger.info(args)

    asyncio.run(main(args))
