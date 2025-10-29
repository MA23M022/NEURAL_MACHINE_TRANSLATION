from pydantic import BaseModel, Field
from typing import Annotated


class userInput(BaseModel):
    ben_sentence_id : Annotated[int, Field(..., gt=0, lt=40000, description="Enter a id of bengali sentece", examples=[10, 23])]