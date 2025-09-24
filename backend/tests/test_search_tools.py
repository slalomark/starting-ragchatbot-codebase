import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test the CourseSearchTool execute method outputs"""

    def test_successful_search_with_results(self, mock_vector_store, sample_search_results):
        """Test successful search that returns results"""
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("Python variables")

        assert "Python Programming Basics" in result
        assert "Lesson 1" in result
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0] == "Python Programming Basics - Lesson 1"

    def test_search_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test search with course name filter"""
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("variables", course_name="Python Programming")

        mock_vector_store.search.assert_called_once_with(
            query="variables",
            course_name="Python Programming",
            lesson_number=None
        )
        assert "Python Programming Basics" in result

    def test_search_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test search with lesson number filter"""
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("variables", course_name="Python Programming", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="variables",
            course_name="Python Programming",
            lesson_number=2
        )

    def test_empty_search_results(self, mock_vector_store, empty_search_results):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("nonexistent topic")

        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0

    def test_search_error_handling(self, mock_vector_store, error_search_results):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("test query")

        assert result == "Test error message"

    def test_tool_definition_structure(self, mock_vector_store):
        """Test that tool definition has required structure"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]


class TestToolManager:
    """Test the ToolManager functionality"""

    def test_tool_registration(self, mock_vector_store):
        """Test registering a tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools
        definitions = manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_tool_execution(self, mock_vector_store, sample_search_results):
        """Test executing a tool through the manager"""
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert "Python Programming Basics" in result

    def test_source_tracking(self, mock_vector_store, sample_search_results):
        """Test source tracking and retrieval"""
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()
        metadata = manager.get_last_source_metadata()

        assert len(sources) == 2
        assert len(metadata) == 2
        assert sources[0] == "Python Programming Basics - Lesson 1"

    def test_source_reset(self, mock_vector_store, sample_search_results):
        """Test resetting sources after retrieval"""
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()

        assert manager.get_last_sources() == []
        assert manager.get_last_source_metadata() == []