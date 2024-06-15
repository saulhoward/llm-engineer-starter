from typing import Optional, List
from pathlib import Path

from .pdf import pdf_text
from .models import MedicalRecord
from .llm import LLMApi
from .utils import read_yaml

from pydantic import BaseModel, Field
from pydantic_core import from_json

import dotenv

dotenv.load_dotenv()


class SplitArticleResponse(BaseModel):
    content: Optional[str] = None
    timestamp: Optional[str] = None


class SplitResponse(BaseModel):
    articles: List[SplitArticleResponse] = Field(
        description="List of articles in the document"
    )


def extract_from_pdf(filepath: str) -> MedicalRecord:
    """
    Entry point.
    """

    # get all text from the PDF
    doc_text = pdf_text(Path(filepath))

    llm_api = LLMApi(api_type="openai")

    # detect events in the document

    split_events_template = read_yaml("./src/prompts/split-events.yaml")["content"]
    system_instructions = split_events_template.format(DOC_TEXT=doc_text)
    rsp = llm_api.chat_completion(
        messages=[{"role": "system", "content": system_instructions}]
    )
    articles = SplitResponse.model_validate(
        from_json(rsp["content"].strip(), allow_partial=True)
    )
    print(articles.model_dump_json())

    record = MedicalRecord(
        id="1",
        content=doc_text,
        events=[],
    )
    return record
