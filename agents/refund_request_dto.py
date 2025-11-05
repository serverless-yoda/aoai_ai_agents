from pydantic import BaseModel
from typing import List

class RefundItemDTO(BaseModel):
    item_code: str
    amount: float

class RefundRequestDTO(BaseModel):
    booking_id: str
    customer_email: str
    reason: str
    requests: List[RefundItemDTO]  # list of items, not self-reference