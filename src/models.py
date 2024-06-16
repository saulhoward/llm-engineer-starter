from typing import List, Optional
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field, UUID4


class MedicalEncounter(BaseModel):
    """
    Event represents a single medical encounter in a patient's record.
    """

    content: str
    findings: List[str]
    prescriptions: List[str]
    timestamp: Optional[datetime] = None


class MedicalRecord(BaseModel):
    """
    MedicalRecord represents a patient's entire medical record.
    """

    id: UUID4 = Field(default_factory=uuid4)
    content: str
    encounters: List[MedicalEncounter] = []
