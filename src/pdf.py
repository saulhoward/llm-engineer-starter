import mimetypes
import math
import os
from io import BytesIO
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud.documentai_v1 import Document
from pypdf import PdfReader, PdfWriter
from loguru import logger


class DocumentAI:
    """Wrapper class around GCP's DocumentAI API."""

    def __init__(self) -> None:

        self.client_options = ClientOptions(  # type: ignore
            api_endpoint=f"{os.getenv('GCP_REGION')}-documentai.googleapis.com",
            credentials_file=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
        self.client = documentai.DocumentProcessorServiceClient(
            client_options=self.client_options
        )
        self.processor_name = self.client.processor_path(
            os.getenv("GCP_PROJECT_ID"),
            os.getenv("GCP_REGION"),
            os.getenv("GCP_PROCESSOR_ID"),
        )

    def __call__(
        self, content: bytes, mime_type: str | None = "application/pdf"
    ) -> Document:
        """Convert bytes into a GCP document. Performs full OCR extraction and layout parsing."""
        #
        raw_document = documentai.RawDocument(content=content, mime_type=mime_type)

        # Configure the process request
        request = documentai.ProcessRequest(
            name=self.processor_name, raw_document=raw_document
        )

        result = self.client.process_document(request=request)
        document = result.document

        return document


def pdf_text(file_path: Path) -> str:
    mime_type = mimetypes.guess_type(file_path)[0]

    with open(file_path, "rb") as fh:
        bytes_stream = BytesIO(fh.read())
    reader = PdfReader(bytes_stream)
    num_pages = len(reader.pages)

    document_ai = DocumentAI()

    if num_pages <= 10:
        return document_ai(bytes_stream.read(), mime_type=mime_type)

    # num_full_chunks = math.floor(num_pages / 10)
    # remainder_chunk_size = num_pages % 10
    num_full_chunks = 1
    remainder_chunk_size = 0
    all_text = ""

    def chunk_pages(pages):
        writer = PdfWriter()
        for page in pages:
            writer.add_page(page)
        with BytesIO() as new_bytes_stream:
            writer.write(new_bytes_stream)
            new_bytes_stream.seek(0)
            doc = document_ai(new_bytes_stream.read())
            return doc.text

    for i in range(num_full_chunks):
        end_page = (i + 1) * 10
        start_page = end_page - 10
        logger.info(f"scanning pdf chunk {start_page}:{end_page}")
        all_text += chunk_pages(reader.pages[start_page:end_page])
    if remainder_chunk_size > 0:
        all_text += chunk_pages(reader.pages[:-remainder_chunk_size])

    return all_text


if __name__ == "__main__":

    # Example Usage
    document_ai = DocumentAI()
    document = document_ai(Path("data/sample.pdf"))
