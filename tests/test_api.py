"""
Tests for the FastAPI endpoints.
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from app.api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "ollama_connected" in data

    def test_health_has_required_fields(self, client):
        response = client.get("/health")
        data = response.json()
        assert "active_model" in data
        assert "available_models" in data


class TestModelsEndpoint:
    """Test the /models endpoint."""

    def test_models_returns_200(self, client):
        response = client.get("/models")
        assert response.status_code == 200


class TestAnalyzeEndpoint:
    """Test the /analyze endpoint."""

    def test_reject_non_pdf(self, client):
        """Unsupported file types (e.g. .csv) should be rejected with 415."""
        response = client.post(
            "/analyze",
            files={"file": ("report.csv", b"col1,col2\nval1,val2", "text/csv")},
        )
        assert response.status_code == 415

    @patch("app.api.routes._run_analysis_job")
    def test_accept_pdf(self, mock_run, client):
        """Valid PDF upload should return a job ID."""
        # Create a minimal PDF-like content
        pdf_content = b"%PDF-1.4 fake content"

        response = client.post(
            "/analyze",
            files={"file": ("report.pdf", pdf_content, "application/pdf")},
            data={"company_name": "Test Corp", "quarter": "Q3 2025"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "processing"

    def test_analyze_with_previous_kpis(self, client):
        """Upload with previous quarter KPIs should work."""
        pdf_content = b"%PDF-1.4 fake content"
        prev_kpis = json.dumps({"revenue": "$80B", "net_income": "$18B"})

        with patch("app.api.routes._run_analysis_job"):
            response = client.post(
                "/analyze",
                files={"file": ("report.pdf", pdf_content, "application/pdf")},
                data={
                    "company_name": "Test Corp",
                    "quarter": "Q3 2025",
                    "previous_kpis_json": prev_kpis,
                },
            )

        assert response.status_code == 200


class TestStatusEndpoint:
    """Test the /status/{job_id} endpoint."""

    def test_unknown_job_returns_404(self, client):
        response = client.get("/status/nonexistent-id")
        assert response.status_code == 404

    def test_known_job_returns_status(self, client):
        from app.api.routes import _jobs

        _jobs["test-job-1"] = {
            "status": "processing",
            "progress": 0.5,
            "current_step": "Extracting KPIs",
            "result": None,
            "error": None,
        }

        response = client.get("/status/test-job-1")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["progress"] == 0.5

        # Cleanup
        del _jobs["test-job-1"]


class TestResultsEndpoint:
    """Test the /results/{job_id} endpoint."""

    def test_unknown_job_returns_404(self, client):
        response = client.get("/results/nonexistent-id")
        assert response.status_code == 404

    def test_incomplete_job_returns_202(self, client):
        from app.api.routes import _jobs

        _jobs["test-job-2"] = {
            "status": "processing",
            "progress": 0.3,
            "current_step": "Running agents",
            "result": None,
            "error": None,
        }

        response = client.get("/results/test-job-2")
        assert response.status_code == 202

        # Cleanup
        del _jobs["test-job-2"]
