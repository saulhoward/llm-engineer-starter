from pathlib import Path
from typing import List, Optional

import dotenv
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_core import from_json

from .llm import LLMApi
from .models import MedicalRecord
from .pdf import pdf_text
from .utils import read_yaml

dotenv.load_dotenv()


class SplitArticleResponse(BaseModel):
    content: Optional[str] = None
    timestamp: Optional[str] = None


class SplitResponse(BaseModel):
    articles: List[SplitArticleResponse] = Field(
        description="List of articles in the document"
    )


def detect_event_boundary(llm: LLMApi, lines: List[str]):
    lines_w_numbers = np.array([])
    for i in range(len(lines)):
        lines_w_numbers[i] = f"{i:03} " + lines[i]

    split_events_template = read_yaml("./src/prompts/detect_event_boundary.yaml")[
        "content"
    ]
    system_instructions = split_events_template.format(
        DOC_TEXT="\n".join(lines_w_numbers)
    )
    rsp = llm.chat_completion(
        messages=[{"role": "system", "content": system_instructions}],
        response_format=None,
    )
    print(rsp["content"].strip())
    # articles = SplitResponse.model_validate(
    #     from_json(rsp["content"].strip(), allow_partial=True)
    # )


def extract_from_pdf(filepath: str) -> MedicalRecord:
    """
    Entry point.
    """
    doc_lines = []

    # # get all text from the PDF using docAI
    # doc_text = pdf_text(Path(filepath))
    # doc_lines = doc_text.splitlines()

    # use the cached data
    with open("./data/pdf-text.txt") as f:
        doc_lines = f.read().splitlines()

    logger.info(f"found record [lines={len(doc_lines)}]")

    # init client
    llm_api = LLMApi(api_type="openai")

    # # send windows of the lines, looking for event boundaries
    # chunk_size = 100
    # start = 0
    # end = len(doc_lines)

    # detect events in the document
    detect_event_boundary(llm_api, doc_lines[0:100])

    # split_events_template = read_yaml("./src/prompts/split-events.yaml")["content"]
    # system_instructions = split_events_template.format(DOC_TEXT=doc_text)
    # rsp = llm_api.chat_completion(
    #     messages=[{"role": "system", "content": system_instructions}]
    # )
    # articles = SplitResponse.model_validate(
    #     from_json(rsp["content"].strip(), allow_partial=True)
    # )
    # print(articles.model_dump_json())

    record = MedicalRecord(
        id="1",
        content="",
        events=[],
    )
    return record
