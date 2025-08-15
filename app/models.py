from pydantic import BaseModel

class QuantityRequest(BaseModel):
    quantity: int
