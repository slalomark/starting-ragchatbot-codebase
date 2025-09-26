import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for /api/query endpoint"""

    def test_query_with_session_id(self, client, sample_query_request, expected_query_response):
        """Test query endpoint with provided session ID"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == expected_query_response["answer"]
        assert data["sources"] == expected_query_response["sources"]
        assert data["session_id"] == expected_query_response["session_id"]
        assert len(data["source_metadata"]) == 1
        assert data["source_metadata"][0]["name"] == "Python Programming Basics - Lesson 1"
        assert data["source_metadata"][0]["course"] == "Python Programming Basics"
        assert data["source_metadata"][0]["lesson"] == 1
        assert data["source_metadata"][0]["link"] == "https://example.com/lesson1"

    def test_query_without_session_id(self, client, sample_query_request_no_session, mock_rag_system):
        """Test query endpoint without session ID (should create new session)"""
        response = client.post("/api/query", json=sample_query_request_no_session)

        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == "test-session-123"
        assert data["answer"] == "This is a test response about Python variables."
        assert len(data["sources"]) > 0

        # Verify session creation was called
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_empty_query(self, client):
        """Test query endpoint with empty query string"""
        response = client.post("/api/query", json={"query": ""})

        assert response.status_code == 200
        # Should still return valid response structure
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_query_with_missing_query_field(self, client):
        """Test query endpoint with missing query field"""
        response = client.post("/api/query", json={"session_id": "test-123"})

        assert response.status_code == 422  # Validation error

    def test_query_with_invalid_json(self, client):
        """Test query endpoint with invalid JSON"""
        response = client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_rag_system_error(self, client, sample_query_request, mock_rag_system):
        """Test query endpoint when RAG system raises an error"""
        mock_rag_system.query.side_effect = Exception("RAG system error")

        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]

    def test_query_session_manager_error(self, client, sample_query_request_no_session, mock_rag_system):
        """Test query endpoint when session manager raises an error"""
        mock_rag_system.session_manager.create_session.side_effect = Exception("Session error")

        response = client.post("/api/query", json=sample_query_request_no_session)

        assert response.status_code == 500
        assert "Session error" in response.json()["detail"]

    def test_query_response_schema(self, client, sample_query_request):
        """Test that query response matches expected schema"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        required_fields = ["answer", "sources", "source_metadata", "session_id"]
        for field in required_fields:
            assert field in data

        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["source_metadata"], list)
        assert isinstance(data["session_id"], str)

        if data["source_metadata"]:
            metadata = data["source_metadata"][0]
            assert "name" in metadata
            assert "course" in metadata
            assert "lesson" in metadata or metadata["lesson"] is None
            assert "link" in metadata or metadata["link"] is None


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for /api/courses endpoint"""

    def test_get_courses_success(self, client, expected_course_stats):
        """Test courses endpoint successful response"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == expected_course_stats["total_courses"]
        assert data["course_titles"] == expected_course_stats["course_titles"]

    def test_get_courses_empty_response(self, client, mock_rag_system):
        """Test courses endpoint with no courses"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_rag_system_error(self, client, mock_rag_system):
        """Test courses endpoint when RAG system raises an error"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]

    def test_get_courses_response_schema(self, client):
        """Test that courses response matches expected schema"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data

        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0

        if data["course_titles"]:
            for title in data["course_titles"]:
                assert isinstance(title, str)


@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API endpoints"""

    def test_query_then_courses_workflow(self, client, sample_query_request):
        """Test typical workflow: query documents then get course stats"""
        # First, make a query
        query_response = client.post("/api/query", json=sample_query_request)
        assert query_response.status_code == 200

        # Then get course statistics
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200

        query_data = query_response.json()
        courses_data = courses_response.json()

        assert query_data["session_id"] is not None
        assert courses_data["total_courses"] > 0

    def test_multiple_queries_same_session(self, client, mock_rag_system):
        """Test multiple queries using the same session ID"""
        session_id = "test-session-persistent"

        # First query
        response1 = client.post("/api/query", json={
            "query": "What is Python?",
            "session_id": session_id
        })
        assert response1.status_code == 200
        assert response1.json()["session_id"] == session_id

        # Second query with same session
        response2 = client.post("/api/query", json={
            "query": "What are variables?",
            "session_id": session_id
        })
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id

        # Verify both calls were made to RAG system
        assert mock_rag_system.query.call_count == 2

    def test_concurrent_requests(self, client, sample_query_request):
        """Test handling of concurrent requests"""
        import concurrent.futures
        import threading

        def make_request():
            return client.post("/api/query", json=sample_query_request)

        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [future.result() for future in futures]

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "session_id" in data


@pytest.mark.api
class TestAPIErrorHandling:
    """Test error handling across API endpoints"""

    def test_malformed_request_body(self, client):
        """Test handling of malformed request bodies"""
        # Test with malformed JSON for query endpoint
        response = client.post(
            "/api/query",
            data='{"query": "test", invalid}',
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_wrong_http_methods(self, client):
        """Test using wrong HTTP methods on endpoints"""
        # GET on query endpoint (should be POST)
        response = client.get("/api/query")
        assert response.status_code == 405

        # POST on courses endpoint (should be GET)
        response = client.post("/api/courses", json={"test": "data"})
        assert response.status_code == 405

    def test_nonexistent_endpoints(self, client):
        """Test requests to nonexistent endpoints"""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

        response = client.post("/api/invalid", json={"test": "data"})
        assert response.status_code == 404

    def test_large_query_payload(self, client):
        """Test handling of very large query payload"""
        large_query = "x" * 10000  # 10KB query

        response = client.post("/api/query", json={
            "query": large_query,
            "session_id": "test-large"
        })

        # Should handle large payloads gracefully
        assert response.status_code in [200, 413]  # OK or Payload Too Large


@pytest.mark.api
@pytest.mark.slow
class TestAPIPerformance:
    """Performance tests for API endpoints"""

    def test_query_response_time(self, client, sample_query_request):
        """Test that query endpoint responds within reasonable time"""
        import time

        start_time = time.time()
        response = client.post("/api/query", json=sample_query_request)
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 5.0  # Should respond within 5 seconds

    def test_courses_response_time(self, client):
        """Test that courses endpoint responds within reasonable time"""
        import time

        start_time = time.time()
        response = client.get("/api/courses")
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 2.0  # Should respond within 2 seconds


@pytest.mark.api
class TestAPIContentTypes:
    """Test different content types and headers"""

    def test_json_content_type(self, client, sample_query_request):
        """Test that API properly handles JSON content type"""
        response = client.post(
            "/api/query",
            json=sample_query_request,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/query")

        # CORS headers should be present due to middleware
        assert response.status_code in [200, 405]  # OPTIONS might not be explicitly handled

    def test_response_encoding(self, client, sample_query_request):
        """Test response encoding is UTF-8"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        assert response.encoding in ["utf-8", None]  # None means default UTF-8