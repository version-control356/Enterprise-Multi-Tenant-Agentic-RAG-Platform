"""Unit tests for safe, dependency-local document parsing behavior."""

import unittest

from app.ingestion.parser import UniversalDocumentParser


class DocumentParserTests(unittest.TestCase):
    """Verify supported content and invalid-input handling."""

    def test_extracts_utf8_text(self) -> None:
        """Extract text uploaded as a plain-text document."""
        self.assertEqual(
            UniversalDocumentParser.extract_text(b"tenant-safe knowledge", "notes.txt"),
            "tenant-safe knowledge",
        )

    def test_rejects_unsupported_extension(self) -> None:
        """Reject unknown upload types instead of attempting to parse them."""
        with self.assertRaises(ValueError):
            UniversalDocumentParser.extract_text(b"binary", "payload.exe")

    def test_chunks_nonempty_document(self) -> None:
        """Create bounded chunks from a text document."""
        chunks = UniversalDocumentParser.chunk_document("word " * 400, "notes.txt")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))
