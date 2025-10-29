from pydantic import BaseModel, Field
from typing import Annotated


class modelOutput(BaseModel):
    ben_sentence : Annotated[str, Field(..., description = "The bengali sentence", examples=["দুইটি কুকুর বরফের মধ্যে লড়াই করছ"])]
    actual_eng_sentence : Annotated[str, Field(..., description="Actual english sentence",
                                                examples=["two dogs are fighting in the snow"])]
    pred_eng_sentence : Annotated[str, Field(..., description="Predicted english sentence",
                                                examples=["two dogs are running through the grass"])]

