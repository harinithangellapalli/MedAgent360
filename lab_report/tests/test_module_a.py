"""
MedAgent 360 · Module A · Tests
Unit tests for pdf_parser, vector_store, rag_pipeline, and voice modules.
Run with: pytest lab_report/tests/ -v
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


# ── PDF Parser Tests ──────────────────────────────────────────────────────────

class TestPDFParser:
    def test_file_not_found(self):
        from lab_report.pdf_parser import extract_lab_values
        with pytest.raises(FileNotFoundError):
            extract_lab_values("/nonexistent/report.pdf")

    def test_parse_numeric_value(self):
        from lab_report.rag_pipeline import _parse_numeric
        assert _parse_numeric("13.5") == 13.5
        assert _parse_numeric("13.5 g/dL") == 13.5
        assert _parse_numeric("N/A") is None
        assert _parse_numeric("") is None

    def test_extract_patient_info(self):
        from lab_report.pdf_parser import _extract_patient_info
        text = "Patient Name: Rajesh Kumar\nAge: 45 Yrs\nDate: 27/02/2026"
        info = _extract_patient_info(text)
        assert info["age"] == "45"
        assert "27/02/2026" in info["date"]


# ── Classification Tests ──────────────────────────────────────────────────────

class TestClassification:
    def test_normal_classification(self):
        from lab_report.rag_pipeline import _classify
        assert _classify(14.0, 13.0, 17.0) == "NORMAL"

    def test_high_classification(self):
        from lab_report.rag_pipeline import _classify
        assert _classify(18.0, 13.0, 17.0) == "HIGH"

    def test_low_classification(self):
        from lab_report.rag_pipeline import _classify
        assert _classify(10.0, 13.0, 17.0) == "LOW"

    def test_critical_very_high(self):
        from lab_report.rag_pipeline import _classify
        # > 1.5x max → CRITICAL
        assert _classify(30.0, 13.0, 17.0) == "CRITICAL"

    def test_critical_very_low(self):
        from lab_report.rag_pipeline import _classify
        # < 0.7x min → CRITICAL
        assert _classify(5.0, 13.0, 17.0) == "CRITICAL"


# ── Vector Store Tests ────────────────────────────────────────────────────────

class TestVectorStore:
    def test_benchmark_data_not_empty(self):
        from lab_report.vector_store import MEDICAL_BENCHMARKS
        assert len(MEDICAL_BENCHMARKS) > 10

    def test_benchmark_structure(self):
        from lab_report.vector_store import MEDICAL_BENCHMARKS
        for bench in MEDICAL_BENCHMARKS:
            assert "test" in bench
            assert "min" in bench
            assert "max" in bench
            assert "unit" in bench
            assert bench["min"] <= bench["max"], f"Invalid range for {bench['test']}"

    def test_build_vector_store_creates_collection(self, tmp_path):
        from lab_report.vector_store import build_vector_store
        collection = build_vector_store(persist_path=str(tmp_path / "test_chroma"))
        assert collection is not None
        assert collection.count() > 0

    def test_query_hemoglobin(self, tmp_path):
        from lab_report.vector_store import build_vector_store, query_benchmark
        collection = build_vector_store(persist_path=str(tmp_path / "test_chroma"))
        results = query_benchmark(collection, "Hemoglobin")
        assert len(results) > 0
        assert "min" in results[0]
        assert "max" in results[0]


# ── Voice Tests ───────────────────────────────────────────────────────────────

class TestVoice:
    @patch("lab_report.voice.gTTS")
    def test_generate_audio_english(self, mock_gtts, tmp_path):
        mock_instance = MagicMock()
        mock_gtts.return_value = mock_instance

        from lab_report.voice import generate_audio
        output = generate_audio("Test summary", "English", str(tmp_path / "test.mp3"))

        mock_gtts.assert_called_once_with(text="Test summary", lang="en", slow=False)
        mock_instance.save.assert_called_once()

    @patch("lab_report.voice.gTTS")
    def test_generate_audio_hindi(self, mock_gtts, tmp_path):
        mock_instance = MagicMock()
        mock_gtts.return_value = mock_instance

        from lab_report.voice import generate_audio
        generate_audio("परीक्षण", "Hindi", str(tmp_path / "test_hi.mp3"))
        mock_gtts.assert_called_with(text="परीक्षण", lang="hi", slow=False)

    def test_language_codes_complete(self):
        from lab_report.voice import LANGUAGE_CODES
        assert "English" in LANGUAGE_CODES
        assert "Telugu" in LANGUAGE_CODES
        assert "Hindi" in LANGUAGE_CODES


# ── Integration Smoke Test ────────────────────────────────────────────────────

class TestIntegration:
    def test_classify_lab_values_with_mock_data(self, tmp_path):
        """Test the classify step with synthetic lab values."""
        from lab_report.rag_pipeline import classify_lab_values
        from unittest.mock import patch

        sample_values = [
            {"test": "Hemoglobin", "value": "10.5", "unit": "g/dL", "reference": "13-17"},
            {"test": "Platelets", "value": "200", "unit": "10³/µL", "reference": "150-400"},
            {"test": "TSH", "value": "8.5", "unit": "µIU/mL", "reference": "0.4-4.0"},
        ]

        with patch("lab_report.rag_pipeline.build_vector_store") as mock_store:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 5
            mock_collection.query.return_value = {
                "metadatas": [[{"min": 13.0, "max": 17.0, "unit": "g/dL", "description": "test"}]]
            }
            mock_store.return_value = mock_collection

            result = classify_lab_values(sample_values)
            assert len(result) == 3
            for r in result:
                assert "status" in r
                assert "risk_icon" in r
