import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Config()
    config.AWS_REGION = "us-east-1"
    config.BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    return config


@pytest.fixture
def sample_course():
    """Sample course data for testing"""
    lessons = [
        Lesson(lesson_number=1, title="Introduction to Python", lesson_link="https://example.com/lesson1"),
        Lesson(lesson_number=2, title="Variables and Data Types", lesson_link="https://example.com/lesson2"),
        Lesson(lesson_number=3, title="Control Structures", lesson_link="https://example.com/lesson3")
    ]

    return Course(
        title="Python Programming Basics",
        course_link="https://example.com/python-course",
        instructor="John Doe",
        lessons=lessons
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    chunks = [
        CourseChunk(
            content="Course Python Programming Basics Lesson 1 content: Python is a powerful programming language used for web development, data science, and automation.",
            course_title="Python Programming Basics",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Course Python Programming Basics Lesson 1 content: Python was created by Guido van Rossum and first released in 1991.",
            course_title="Python Programming Basics",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Course Python Programming Basics Lesson 2 content: Variables in Python are used to store data values. Python has different data types including strings, integers, and floats.",
            course_title="Python Programming Basics",
            lesson_number=2,
            chunk_index=2
        ),
        CourseChunk(
            content="Course Python Programming Basics Lesson 3 content: Control structures like if statements and loops allow you to control the flow of your program.",
            course_title="Python Programming Basics",
            lesson_number=3,
            chunk_index=3
        )
    ]
    return chunks


@pytest.fixture
def sample_search_results(sample_course_chunks):
    """Sample search results for testing"""
    return SearchResults(
        documents=[chunk.content for chunk in sample_course_chunks[:2]],
        metadata=[
            {
                "course_title": chunk.course_title,
                "lesson_number": chunk.lesson_number,
                "chunk_index": chunk.chunk_index
            } for chunk in sample_course_chunks[:2]
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """Error search results for testing"""
    return SearchResults.empty("Test error message")


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock()
    mock_store.search = Mock()
    mock_store.get_lesson_link = Mock(return_value="https://example.com/lesson1")
    mock_store.add_course_metadata = Mock()
    mock_store.add_course_content = Mock()
    mock_store.get_existing_course_titles = Mock(return_value=["Python Programming Basics"])
    mock_store.get_course_count = Mock(return_value=1)
    return mock_store


@pytest.fixture
def mock_bedrock_client():
    """Mock AWS Bedrock client for testing"""
    mock_client = Mock()

    # Mock successful response without tools
    mock_response_body = {
        "content": [{"text": "This is a test response from Claude."}],
        "stop_reason": "end_turn"
    }

    mock_response = {
        "body": Mock()
    }
    mock_response["body"].read.return_value = Mock()
    mock_response["body"].read.return_value = str.encode(str(mock_response_body).replace("'", '"'))

    mock_client.invoke_model.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_tool_use_response():
    """Mock Bedrock response that includes tool use"""
    return {
        "content": [
            {
                "type": "tool_use",
                "id": "tool_12345",
                "name": "search_course_content",
                "input": {
                    "query": "Python variables",
                    "course_name": "Python Programming"
                }
            }
        ],
        "stop_reason": "tool_use"
    }


@pytest.fixture
def mock_final_response():
    """Mock final response after tool execution"""
    return {
        "content": [{"text": "Based on the search results, variables in Python are used to store data values."}],
        "stop_reason": "end_turn"
    }


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    mock_manager = Mock()
    mock_manager.get_conversation_history = Mock(return_value=None)
    mock_manager.add_exchange = Mock()
    return mock_manager


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager for testing"""
    mock_manager = Mock()
    mock_manager.get_tool_definitions = Mock(return_value=[
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    ])
    mock_manager.execute_tool = Mock(return_value="Mocked search results")
    mock_manager.get_last_sources = Mock(return_value=["Python Programming Basics - Lesson 1"])
    mock_manager.get_last_source_metadata = Mock(return_value=[
        {
            "name": "Python Programming Basics - Lesson 1",
            "course": "Python Programming Basics",
            "lesson": 1,
            "link": "https://example.com/lesson1"
        }
    ])
    mock_manager.reset_sources = Mock()
    return mock_manager


@pytest.fixture
def course_search_test_cases():
    """Test cases for CourseSearchTool testing"""
    return [
        {
            "name": "successful_search_with_results",
            "query": "Python variables",
            "course_name": None,
            "lesson_number": None,
            "expected_results": True,
            "expected_error": None
        },
        {
            "name": "successful_search_with_course_filter",
            "query": "Python basics",
            "course_name": "Python Programming",
            "lesson_number": None,
            "expected_results": True,
            "expected_error": None
        },
        {
            "name": "successful_search_with_lesson_filter",
            "query": "variables",
            "course_name": "Python Programming",
            "lesson_number": 2,
            "expected_results": True,
            "expected_error": None
        },
        {
            "name": "empty_results",
            "query": "nonexistent topic",
            "course_name": None,
            "lesson_number": None,
            "expected_results": False,
            "expected_error": None
        },
        {
            "name": "invalid_course",
            "query": "anything",
            "course_name": "Nonexistent Course",
            "lesson_number": None,
            "expected_results": False,
            "expected_error": "No course found matching 'Nonexistent Course'"
        },
        {
            "name": "search_error",
            "query": "test query",
            "course_name": None,
            "lesson_number": None,
            "expected_results": False,
            "expected_error": "Search error: Test database error"
        }
    ]


@pytest.fixture
def ai_generator_test_cases():
    """Test cases for AIGenerator testing"""
    return [
        {
            "name": "simple_query_no_tools",
            "query": "What is 2+2?",
            "tools": None,
            "expected_tool_use": False,
            "expected_response_contains": "4"
        },
        {
            "name": "course_content_query_with_tools",
            "query": "Tell me about Python variables",
            "tools": [{"name": "search_course_content"}],
            "expected_tool_use": True,
            "expected_response_contains": "variables"
        },
        {
            "name": "general_programming_question",
            "query": "What are the benefits of object-oriented programming?",
            "tools": [{"name": "search_course_content"}],
            "expected_tool_use": False,
            "expected_response_contains": "object-oriented"
        }
    ]


# API Testing Fixtures

@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    mock_rag = Mock()

    # Mock session manager
    mock_session_manager = Mock()
    mock_session_manager.create_session.return_value = "test-session-123"
    mock_rag.session_manager = mock_session_manager

    # Mock query method
    mock_rag.query.return_value = (
        "This is a test response about Python variables.",
        ["Python Programming Basics - Lesson 1"],
        [
            {
                "name": "Python Programming Basics - Lesson 1",
                "course": "Python Programming Basics",
                "lesson": 1,
                "link": "https://example.com/lesson1"
            }
        ]
    )

    # Mock course analytics
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Programming Basics", "Advanced Python"]
    }

    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create test FastAPI app without static file mounting issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Create test app
    app = FastAPI(title="Test Course Materials RAG System")

    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Pydantic models (inline to avoid import issues)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceMetadata(BaseModel):
        name: str
        course: str
        lesson: Optional[int] = None
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        source_metadata: List[SourceMetadata]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources, source_metadata = mock_rag_system.query(request.query, session_id)
            metadata_models = [SourceMetadata(**meta) for meta in source_metadata]

            return QueryResponse(
                answer=answer,
                sources=sources,
                source_metadata=metadata_models,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def client(test_app):
    """Create test client for API testing"""
    from fastapi.testclient import TestClient
    return TestClient(test_app)


@pytest.fixture
async def async_client(test_app):
    """Create async test client for API testing"""
    from httpx import AsyncClient
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_query_request():
    """Sample query request for API testing"""
    return {
        "query": "What are Python variables?",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_query_request_no_session():
    """Sample query request without session ID"""
    return {
        "query": "What are Python data types?"
    }


@pytest.fixture
def expected_query_response():
    """Expected query response for API testing"""
    return {
        "answer": "This is a test response about Python variables.",
        "sources": ["Python Programming Basics - Lesson 1"],
        "source_metadata": [
            {
                "name": "Python Programming Basics - Lesson 1",
                "course": "Python Programming Basics",
                "lesson": 1,
                "link": "https://example.com/lesson1"
            }
        ],
        "session_id": "test-session-123"
    }


@pytest.fixture
def expected_course_stats():
    """Expected course statistics response"""
    return {
        "total_courses": 2,
        "course_titles": ["Python Programming Basics", "Advanced Python"]
    }


# Utility functions for test setup
def setup_mock_chroma_results(documents: List[str], metadata: List[Dict], distances: List[float]):
    """Helper to create mock ChromaDB results"""
    return {
        'documents': [documents],
        'metadatas': [metadata],
        'distances': [distances]
    }


def setup_mock_boto3():
    """Helper to patch boto3 client creation"""
    return patch('boto3.client')