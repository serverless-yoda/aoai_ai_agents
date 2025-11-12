import asyncio
import requests
from pydantic import BaseModel
from typing import List

from config import settings

from openai import AsyncAzureOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    ModelSettings,
    set_tracing_disabled
)

set_tracing_disabled(True)

class Crypto(BaseModel):
    coin_ids: List[str]

@function_tool
def get_crypto_prices(crypto:Crypto)-> str:
    ids = ",".join(crypto.coin_ids)
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data

@function_tool
def get_price() -> str:
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    response = requests.get(url)
    price = response.json()["bitcoin"]["usd"]
    return f"${price:,.2f} USD."


async def main() -> None:
    async with AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_version,
        azure_endpoint=str(settings.azure_endpoint_url)
    )  as client:

        model = OpenAIChatCompletionsModel(
            model=settings.azure_deployment_name,
            openai_client=client
        )

        crypto_agent=Agent(
            name="cryptonAgent",
            instructions="You are a crypto assistant. Use tools to get real-time data. When getting cryptocurrency prices, call the tool only once for all requests.",
            model=model,
            tools=[get_crypto_prices]
        )

        result = await Runner.run(crypto_agent, "What is the price of bitcoin and ethereum")
        print(result.final_output)


asyncio.run(main())