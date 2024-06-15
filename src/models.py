from typing import List
from datetime import datetime

from pydantic import BaseModel


class Event(BaseModel):
    """
    Event represents a single medical event in a patient's record.
    """

    content: str
    timestamp: datetime


class MedicalRecord(BaseModel):
    """
    MedicalRecord represents a patient's entire medical record.
    """

    id: str
    content: str
    events: List[Event]
