from unittest.mock import Mock, patch

from rag_system import RAGSystem


class TestRAGSystemContentQueries:
    """Test how the RAG system handles content-query related questions"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_content_query_workflow(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test end-to-end content query processing"""
        # Setup mocks
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = (
            "Variables in Python store data values."
        )
        mock_ai_gen.return_value = mock_ai_instance

        mock_vector_instance = Mock()
        mock_vector_store.return_value = mock_vector_instance

        mock_session_instance = Mock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session_mgr.return_value = mock_session_instance

        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = ["Python Basics - Lesson 2"]
        mock_tool_manager.get_last_source_metadata.return_value = [
            {"name": "Python Basics - Lesson 2", "course": "Python Basics", "lesson": 2}
        ]

        # Create RAG system
        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager = mock_tool_manager

        # Test query
        response, sources, metadata = rag_system.query("What are Python variables?")

        # Verify workflow
        assert response == "Variables in Python store data values."
        assert sources == ["Python Basics - Lesson 2"]
        assert len(metadata) == 1
        mock_ai_instance.generate_response.assert_called_once()
        mock_tool_manager.get_last_sources.assert_called_once()
        mock_tool_manager.reset_sources.assert_called_once()

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_session_management_in_queries(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test session management during queries"""
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Test response"
        mock_ai_gen.return_value = mock_ai_instance

        mock_session_instance = Mock()
        mock_session_instance.get_conversation_history.return_value = (
            "Previous: context"
        )
        mock_session_mgr.return_value = mock_session_instance

        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = []
        mock_tool_manager.get_last_source_metadata.return_value = []

        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager = mock_tool_manager

        # Test with session ID
        response, sources, metadata = rag_system.query(
            "Follow-up question", session_id="test_session"
        )

        # Verify session interactions
        mock_session_instance.get_conversation_history.assert_called_once_with(
            "test_session"
        )
        mock_session_instance.add_exchange.assert_called_once_with(
            "test_session", "Follow-up question", "Test response"
        )

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_tool_definitions_passed_to_ai(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test that tool definitions are passed to AI generator"""
        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance

        mock_tool_manager = Mock()
        mock_tool_definitions = [
            {"name": "search_course_content", "description": "Search courses"}
        ]
        mock_tool_manager.get_tool_definitions.return_value = mock_tool_definitions
        mock_tool_manager.get_last_sources.return_value = []
        mock_tool_manager.get_last_source_metadata.return_value = []

        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager = mock_tool_manager

        rag_system.query("Test query")

        # Verify tools passed to AI
        call_args = mock_ai_instance.generate_response.call_args
        assert call_args[1]["tools"] == mock_tool_definitions
        assert call_args[1]["tool_manager"] == mock_tool_manager

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_source_reset_after_query(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test that sources are reset after each query"""
        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance

        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = ["Test Source"]
        mock_tool_manager.get_last_source_metadata.return_value = [{"name": "Test"}]

        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager = mock_tool_manager

        rag_system.query("Test query")

        # Verify sources retrieved and then reset
        mock_tool_manager.get_last_sources.assert_called_once()
        mock_tool_manager.get_last_source_metadata.assert_called_once()
        mock_tool_manager.reset_sources.assert_called_once()

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_without_session(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test query processing without session ID"""
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "No session response"
        mock_ai_gen.return_value = mock_ai_instance

        mock_session_instance = Mock()
        mock_session_mgr.return_value = mock_session_instance

        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = []
        mock_tool_manager.get_last_source_metadata.return_value = []

        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager = mock_tool_manager

        response, sources, metadata = rag_system.query("Test query")

        # Verify no session interactions
        mock_session_instance.get_conversation_history.assert_not_called()
        mock_session_instance.add_exchange.assert_not_called()
        assert response == "No session response"

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_prompt_formatting(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test that user query is properly formatted as prompt"""
        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance

        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = []
        mock_tool_manager.get_last_source_metadata.return_value = []

        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager = mock_tool_manager

        user_query = "What are Python functions?"
        rag_system.query(user_query)

        # Verify prompt formatting
        call_args = mock_ai_instance.generate_response.call_args
        expected_prompt = f"Answer this question about course materials: {user_query}"
        assert call_args[1]["query"] == expected_prompt
