"""CV parser implementation using pypdf and python-docx."""

import io

import structlog

from src.ml_engine.domain.ports import CVParserService
from src.shared.exceptions import MLPipelineError

logger = structlog.get_logger(__name__)


class LocalCVParserService(CVParserService):
    """Extracts text from PDF and DOCX files using local libraries."""

    async def extract_text(self, content: bytes, content_type: str) -> str:
        """Dispatch to the appropriate parser based on content type."""
        if content_type == "application/pdf":
            return self._extract_from_pdf(content)
        elif content_type == (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            return self._extract_from_docx(content)
        else:
            raise MLPipelineError(f"Unsupported content type for parsing: {content_type}")

    def _extract_from_pdf(self, content: bytes) -> str:
        """Extract text from a PDF file using pypdf."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            pages_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

            full_text = "\n".join(pages_text)
            logger.debug("PDF extracted", pages=len(reader.pages), chars=len(full_text))
            return full_text
        except Exception as exc:
            logger.error("PDF extraction failed", error=str(exc))
            raise MLPipelineError("Failed to parse PDF document") from exc

    def _extract_from_docx(self, content: bytes) -> str:
        """Extract text from a DOCX file using python-docx."""
        try:
            from docx import Document

            doc = Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n".join(paragraphs)
            logger.debug("DOCX extracted", paragraphs=len(paragraphs), chars=len(full_text))
            return full_text
        except Exception as exc:
            logger.error("DOCX extraction failed", error=str(exc))
            raise MLPipelineError("Failed to parse DOCX document") from exc
