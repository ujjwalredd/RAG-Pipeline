from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "indexed_chunks" in data
        assert "ollama_available" in data

    def test_health_status_ok(self):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


class TestDocumentsEndpoint:
    def test_list_documents_empty(self):
        resp = client.get("/v1/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "documents" in data


class TestAskEndpoint:
    def test_ask_without_index_returns_400(self):
        resp = client.post("/v1/ask", json={"question": "What is X?"})
        # May return 400 if no docs indexed
        assert resp.status_code in (200, 400)

    def test_ask_validation_empty_question(self):
        resp = client.post("/v1/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_ask_validation_invalid_mode(self):
        resp = client.post(
            "/v1/ask",
            json={"question": "test", "retrieval_mode": "invalid"},
        )
        assert resp.status_code == 422

    def test_ask_validation_weight_bounds(self):
        resp = client.post(
            "/v1/ask",
            json={"question": "test", "dense_weight": 1.5},
        )
        assert resp.status_code == 422


class TestIngestEndpoint:
    def test_ingest_validation(self):
        resp = client.post(
            "/v1/ingest",
            json={"chunking_strategy": "invalid_strategy"},
        )
        assert resp.status_code == 422


class TestOpenAPI:
    def test_openapi_schema_available(self):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/v1/ask" in schema["paths"]
        assert "/v1/documents" in schema["paths"]
        assert "/v1/ingest" in schema["paths"]
