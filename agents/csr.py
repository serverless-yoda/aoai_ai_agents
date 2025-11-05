from __future__ import annotations

import asyncio
from pathlib import PurePath
import random
import sys
from typing import Final, Union, List

from refund_request_dto import RefundRequestDTO
from openai import AsyncAzureOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    function_tool,
    ModelSettings,
    StopAtTools
)
from config import settings

# Disable Agents SDK tracing to avoid OpenAI (non‑Azure) calls
set_tracing_disabled(True)

# Windows: avoid Proactor teardown warnings
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Demo status sets (replace with randomized generator if desired)
_BOOKING_STATUS_INQUIRY: Final[set[int]] = {100, 200}
_BOOKING_STATUS_CONFIRMED: Final[set[int]] = {300, 400}
_DEFAULT_STATUS: Final[str] = "Unknown"
_BOOKING_ID_POOL: Final[tuple[int,...]]= (100,200,300,400)

_PURPOSE_POOL: Final[tuple[str, ...]] = (
    "I want to request for invoice.",
    "I want to do cancellation.",
    "I want to inquire about the status.",
    "I want to ask to move the pickup date.",
    "I want to ask if there are some promos I can grab.",
    "I want to upgrade my campervan.",
    "I want to add extra insurance coverage.",
    "I want to change the drop-off location.",
    "I want to extend my rental period.",
    "I want to ask about payment options.",
    "I want to report an issue with my booking.",
    "I want to confirm if pets are allowed.",
    "I want to request a child seat.",
    "I want to ask about fuel policy.",
    "I want to check if I can pick up earlier.",
    "I want to ask about cancellation fees.",
    "I want to ask refund.My email is manuel.agbayani@thlonline.com. Customer id is 999. Its is pricey for me.",
    
)

@function_tool
def get_rental_invoice(booking_id: str, amount: str) -> str:
    return f"{booking_id} invoice is {amount}"

@function_tool(name_override="get_booking_status")
def get_booking_status(booking_id: Union[int, str]) -> str:
    """
    Return the status for a campervan booking.

    Args:
        booking_id: The booking identifier (int or string containing digits).

    Returns:
        One of: "Inquiry", "Confirmed", or "Unknown".
    """
    # Accept either int or str and normalize to int
    try:
        bid = int(booking_id)
    except (TypeError, ValueError):
        return _DEFAULT_STATUS

    if bid in _BOOKING_STATUS_INQUIRY:
        return "Inquiry"
    if bid in _BOOKING_STATUS_CONFIRMED:
        return "Confirmed"
    return _DEFAULT_STATUS

@function_tool
def process_rental_refund(request: RefundRequestDTO) -> str:
    messages = (
        f"Refund for booking id: {request.booking_id} will be send to {request.customer_email}, Reason specified is {request.reason}"
    )
    return messages

async def main() -> None:

    purpose = random.choice(_PURPOSE_POOL)
    booking_number = f"My booking number is {random.choice(_BOOKING_ID_POOL)}"
    user_context= f'{purpose}{booking_number}'

    print("\n===============================")
    print(f'User context: {user_context}')
    print("\n===============================")

    async with AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_version,
        azure_endpoint=str(settings.azure_endpoint_url),
    ) as client:
        model = OpenAIChatCompletionsModel(
            model=settings.azure_deployment_name,
            openai_client=client,
        )

        refund_agent = Agent(
            name="Rental Refund Agent",
            instructions=(
                "You are an AI agent that responds to customers who want to refund "
                "their booking payment. Be very courteous" ),
            model=model,
            tools=['process_rental_refund'],
            tool_use_behavior=StopAtTools(stop_at_tool_names=["process_rental_refund"]),
            model_settings=ModelSettings(tool_choice="required")
        )

        retention_agent = Agent(
            name="Retention Customer Agent",
            instructions=(
                "You are an AI agent that responds to customers who want to cancel "
                "their booking. Be very courteous, "
                "relatable, and kind. Offer discounts up to 10% if it helps."
            ),
            model=model
        )

        invoice_agent = Agent(
            name="Invoicing Agent",
            instructions="Generate and return an invoice when requested",
            model=model,
            tools=[get_rental_invoice],
            #stop=StopAtTools.stop_at_tool_names['get_rental_invoice']
            tool_use_behavior=StopAtTools(stop_at_tool_names=["get_rental_invoice"]),
        )

        csr_agent = Agent(
            name="CSR",
            instructions=(
                "You help with campervan bookings. When a booking ID is provided, "
                "call get_booking_status before replying or escalating."
            ),
            model=model,
            tools=[get_booking_status],
            handoffs=[retention_agent, invoice_agent, refund_agent],
            model_settings=ModelSettings(tool_choice="required")
        )

        user_message = user_context
        result = await Runner.run(csr_agent, user_message)
        print("\n=== Agent Response ===\n")
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())