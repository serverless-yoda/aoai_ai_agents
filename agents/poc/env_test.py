import asyncio
from openai import AsyncAzureOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel,set_tracing_disabled
from config import settings

set_tracing_disabled(True)

async def main():
    # Use context manager so the client is cleanly closed before loop shutdown
    async with AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_version,
        azure_endpoint=str(settings.azure_endpoint_url),
    ) as client:

        model = OpenAIChatCompletionsModel(
            model=settings.azure_deployment_name,
            openai_client=client,
        )

        agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant",
            model=model,
        )

        result = await Runner.run(agent, "How to create a Drone for  perimeter surveillance")
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main()) 
