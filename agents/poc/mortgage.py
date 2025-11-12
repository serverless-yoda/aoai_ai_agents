import asyncio
from config import settings
from openai import AsyncAzureOpenAI
from agents import (
    Agent, 
    Runner, 
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    function_tool,
    ModelSettings
)

set_tracing_disabled(True)

@function_tool
def calculate_mortgage(
    principal_amount: float,
    annualized_rate: float,
    number_of_years: int
) -> str:
    monthly_rate = (annualized_rate / 100) / 12
    months = number_of_years * 12
    payment = principal_amount * (monthly_rate) / (1 - (1 + monthly_rate) ** -months)
    print(payment)
    return f"${payment:,.2f}."

async def main() -> None:
    async with AsyncAzureOpenAI(
        api_key = settings.azure_openai_api_key,
        api_version = settings.azure_openai_version,
        azure_endpoint = str(settings.azure_endpoint_url)
    ) as client:

        model = OpenAIChatCompletionsModel(
            model= settings.azure_deployment_name,
            openai_client=client
        )

        mortgage_agent = Agent(
            name="mortgage_agent",
            instructions="You are a mortgage assistant",
            model=model,
            tools=[calculate_mortgage],
            tool_use_behavior="stop_on_first_tool",
            model_settings=ModelSettings(tool_choice='required')
        )

        user_content="What is my monthly payments if I borrow $800,000 at 7% interest for 30 years?"
        result = await Runner.run(mortgage_agent,user_content)
        print(result.final_output)


asyncio.run(main())