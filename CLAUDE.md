# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install dependencies (including boto3 for AWS Bedrock)
uv sync

# Add new dependency
uv add package-name

# Run Python commands
uv run python -m module_name
```

### Environment Setup
```bash
# Copy environment template and configure AWS settings
cp .env.example .env
# Edit .env to set AWS_REGION (and optionally BEDROCK_MODEL_ID)

# Configure AWS credentials using one of these methods:
# 1. AWS CLI: aws configure
# 2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# 3. IAM roles (for EC2/ECS/Lambda deployment)
```

### Testing
```bash
# Run all tests
uv run pytest backend/tests/ -v

# Run specific test file
uv run pytest backend/tests/test_search_tools.py -v

# Run tests with coverage
uv run pytest backend/tests/ --cov=backend --cov-report=html

# Run tests in verbose mode with short traceback
uv run pytest backend/tests/ -v --tb=short
```

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials with the following architecture:

### Core Components

**RAG System** (`rag_system.py`) - Main orchestrator that coordinates all components:
- Manages document processing pipeline
- Orchestrates AI generation with tool usage
- Handles session management and conversation history
- Controls search tool integration

**FastAPI Backend** (`app.py`) - Web server that:
- Serves static frontend files from `../frontend`
- Provides `/api/query` endpoint for user queries
- Provides `/api/courses` endpoint for course statistics
- Initializes RAG system and loads documents from `../docs` on startup

**Vector Store** (`vector_store.py`) - ChromaDB integration:
- Two collections: `course_metadata` and `course_content`
- Uses SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
- Supports semantic search with course/lesson filtering
- Persistent storage in `./chroma_db`

**AI Generator** (`ai_generator.py`) - AWS Bedrock Claude integration:
- Uses `anthropic.claude-3-5-sonnet-20241022-v2:0` model via Bedrock
- Boto3 bedrock-runtime client with local AWS credentials
- Tool-based approach: Claude decides when to search vs. direct answer
- System prompt optimized for educational content
- Temperature 0, max 800 tokens

**Search Tools** (`search_tools.py`) - Tool-based search system:
- `CourseSearchTool` with semantic course name matching
- Supports filtering by course name and lesson number
- Integrated with Claude's tool calling capabilities

### Data Flow

1. **Document Processing**: Files in `docs/` → `DocumentProcessor` → `CourseChunk` objects → `VectorStore`
2. **Query Processing**: User query → FastAPI → `RAGSystem.query()` → `AIGenerator` → (optionally) `SearchTools` → response
3. **Session Management**: Conversation history maintained via `SessionManager` with configurable memory depth

### Key Models

- **Course**: Represents a course with title, lessons, and metadata
- **Lesson**: Individual lesson within a course with number and title
- **CourseChunk**: Text chunk for vector storage with course/lesson context
- **SearchResults**: Container for search results with documents, metadata, and distances

### Configuration

All settings centralized in `config.py`:
- AWS region: `us-east-1` (configurable via AWS_REGION)
- Bedrock model: `anthropic.claude-3-5-sonnet-20241022-v2:0` (configurable via BEDROCK_MODEL_ID)
- Chunk size: 800 characters with 100 character overlap
- Max search results: 5
- Conversation history: 2 exchanges
- ChromaDB path: `./chroma_db`

### Frontend Integration

The frontend (`frontend/`) contains vanilla HTML/CSS/JavaScript that:
- Communicates via `/api/query` and `/api/courses` endpoints
- Handles session management client-side
- Renders markdown responses and collapsible source citations
- Supports suggested questions and real-time loading states

## Development Notes

### Adding New Course Documents
- Place documents in `docs/` directory (supports .pdf, .docx, .txt)
- Documents are automatically processed on server startup
- Use `RAGSystem.add_course_folder()` to manually reload
- Each document should contain course title and lesson structure

### Extending Search Capabilities
- New tools can be registered with `ToolManager.register_tool()`
- Tools must implement the `Tool` abstract base class
- AI automatically decides when to use tools based on system prompt

### Session and Memory Management
- Sessions auto-created on first query if not provided
- Conversation history limited by `MAX_HISTORY` setting
- Session state maintained in memory (not persistent across restarts)

### Vector Store Operations
- Two separate collections for metadata vs. content enable flexible querying
- Course titles used as unique identifiers
- Duplicate detection prevents re-processing existing courses
- I'll start server myself, so Claude should never start run run.sh