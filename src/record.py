from pathlib import Path
from typing import List
import re
from uuid import uuid4

import dotenv
from loguru import logger
import dateparser

from .llm import LLMApi
from .models import MedicalRecord, MedicalEncounter
from .pdf import pdf_text
from .utils import read_yaml

dotenv.load_dotenv()


def detect_encounter_boundary_indexes(llm: LLMApi, lines: List[str]) -> List[int]:
    """
    Return indexes of lines in the list that indicate an encounter boundary.
    """

    def detect_boundaries_in_chunk(lines: List[str]) -> List[int]:
        lines_w_numbers = [f"{i:03} {lines[i]}" for i in range(len(lines))]
        detect_encounters_template = read_yaml(
            "./src/prompts/detect_encounter_boundary.yaml"
        )["content"]
        system_instructions = detect_encounters_template.format(
            DOC_TEXT="\n".join(lines_w_numbers)
        )
        rsp = llm.chat_completion(
            messages=[{"role": "system", "content": system_instructions}],
            response_format=None,
        )
        indexes = []
        for rsp_line in rsp["content"].strip().splitlines():
            m = re.match(r"\s*([-+])?\d+", rsp_line)
            if m is not None:
                indexes.append(int(m.group()))
        return indexes

    encounter_boundary_indexes = []
    chunk_size = 100
    start = 0
    while start < len(lines):
        end = start + chunk_size - 1
        if end > (len(lines) - 1):
            end = len(lines) - 1
        chunk_lines = lines[start:end]
        indexes = detect_boundaries_in_chunk(chunk_lines)
        for i in indexes:
            encounter_boundary_indexes.append(i + start)
        start = end + 1
    return encounter_boundary_indexes


def parse_encounter(llm: LLMApi, lines: List[str]) -> MedicalEncounter:
    """
    Return a structured summary of a medical encounter from the provided content.
    """

    # timestamp for this encounter
    get_timestamp_template = read_yaml("./src/prompts/encounter_timestamp.yaml")[
        "content"
    ]
    system_instructions = get_timestamp_template.format(DOC_TEXT="\n".join(lines))
    timestamp_rsp = llm.chat_completion(
        messages=[{"role": "system", "content": system_instructions}],
        response_format=None,
    )
    # timestamp
    timestamp = dateparser.parse(timestamp_rsp["content"].strip())

    # medical findings in the encounter
    list_findings_template = read_yaml("./src/prompts/list_findings.yaml")["content"]
    system_instructions = list_findings_template.format(DOC_TEXT="\n".join(lines))
    findings_rsp = llm.chat_completion(
        messages=[{"role": "system", "content": system_instructions}],
        response_format=None,
    )
    findings = findings_rsp["content"].strip().splitlines()

    # prescriptions in the encounter
    list_prescriptions_template = read_yaml("./src/prompts/list_prescriptions.yaml")[
        "content"
    ]
    system_instructions = list_prescriptions_template.format(DOC_TEXT="\n".join(lines))
    prescriptions_rsp = llm.chat_completion(
        messages=[{"role": "system", "content": system_instructions}],
        response_format=None,
    )
    prescriptions = prescriptions_rsp["content"].strip().splitlines()

    return MedicalEncounter(
        timestamp=timestamp,
        content="\n".join(lines),
        findings=findings,
        prescriptions=prescriptions,
    )


def extract_from_pdf(filepath: str) -> MedicalRecord:
    """
    Entry point. Takes the filepath of a PDF document and returns structured list of Medical Encounters.
    """
    # get all text from the PDF using docAI
    doc_text = pdf_text(Path(filepath))
    # doc_text = Path("./data/pdf-text.txt").read_text()

    doc_lines = doc_text.splitlines()
    logger.info(f"using record [num_lines={len(doc_lines)}]")

    # init client
    llm_api = LLMApi(api_type="openai")

    # detect medical encounters in the document
    encounter_boundary_indexes = detect_encounter_boundary_indexes(llm_api, doc_lines)
    logger.info(
        f"found {len(encounter_boundary_indexes)} medical encounters in the record"
    )
    encounters: List[MedicalEncounter] = []
    for i in range(len(encounter_boundary_indexes)):
        start = encounter_boundary_indexes[i]
        end = (
            encounter_boundary_indexes[-1]
            if i == len(encounter_boundary_indexes) - 1
            else encounter_boundary_indexes[i + 1]
        )
        encounter = parse_encounter(llm_api, doc_lines[start:end])
        logger.info(
            f"parsed medical encounter {i} [num_findings={len(encounter.findings)}][num_prescriptions={len(encounter.prescriptions)}]"
        )
        encounters.append(encounter)

    return MedicalRecord(id=uuid4(), content=doc_text, encounters=encounters)
