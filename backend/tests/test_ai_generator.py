import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test the AIGenerator's integration with CourseSearchTool"""

    @patch('boto3.client')
    def test_simple_query_without_tool_use(self, mock_boto_client):
        """Test direct response without tool usage"""
        # Setup mock response - no tool use
        mock_response_body = {
            "content": [{"text": "This is a direct answer without tools."}],
            "stop_reason": "end_turn"
        }

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(mock_response_body).encode()

        mock_client = Mock()
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client

        generator = AIGenerator("us-east-1", "test-model")
        result = generator.generate_response("What is 2+2?")

        assert result == "This is a direct answer without tools."
        mock_client.invoke_model.assert_called_once()

    @patch('boto3.client')
    def test_course_query_triggers_tool_use(self, mock_boto_client, mock_tool_manager):
        """Test that course-related queries trigger tool usage"""
        # Mock initial response with tool use
        tool_use_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "search_course_content",
                    "input": {
                        "query": "Python variables",
                        "course_name": "Python Programming"
                    }
                }
            ],
            "stop_reason": "tool_use"
        }

        # Mock final response after tool execution
        final_response = {
            "content": [{"text": "Variables in Python are used to store data values."}],
            "stop_reason": "end_turn"
        }

        mock_client = Mock()
        mock_boto_client.return_value = mock_client

        # Setup multiple responses
        responses = [
            {"body": Mock()},  # Tool use response
            {"body": Mock()}   # Final response
        ]
        responses[0]["body"].read.return_value = json.dumps(tool_use_response).encode()
        responses[1]["body"].read.return_value = json.dumps(final_response).encode()

        mock_client.invoke_model.side_effect = responses

        # Setup tool manager
        mock_tool_manager.execute_tool.return_value = "Search results about Python variables"

        generator = AIGenerator("us-east-1", "test-model")
        tools = [{"name": "search_course_content", "description": "Search courses"}]

        result = generator.generate_response(
            "Tell me about Python variables",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify tool execution
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python variables",
            course_name="Python Programming"
        )
        assert result == "Variables in Python are used to store data values."
        assert mock_client.invoke_model.call_count == 2

    @patch('boto3.client')
    def test_tool_call_parameter_extraction(self, mock_boto_client, mock_tool_manager):
        """Test correct parameter extraction for tool calls"""
        tool_use_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_456",
                    "name": "search_course_content",
                    "input": {
                        "query": "control structures",
                        "course_name": "Python Basics",
                        "lesson_number": 3
                    }
                }
            ],
            "stop_reason": "tool_use"
        }

        final_response = {
            "content": [{"text": "Control structures control program flow."}],
            "stop_reason": "end_turn"
        }

        mock_client = Mock()
        mock_boto_client.return_value = mock_client

        responses = [
            {"body": Mock()},
            {"body": Mock()}
        ]
        responses[0]["body"].read.return_value = json.dumps(tool_use_response).encode()
        responses[1]["body"].read.return_value = json.dumps(final_response).encode()

        mock_client.invoke_model.side_effect = responses
        mock_tool_manager.execute_tool.return_value = "Search results about control structures"

        generator = AIGenerator("us-east-1", "test-model")
        tools = [{"name": "search_course_content"}]

        generator.generate_response(
            "What are control structures in lesson 3?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify all parameters passed correctly
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="control structures",
            course_name="Python Basics",
            lesson_number=3
        )

    @patch('boto3.client')
    def test_conversation_history_inclusion(self, mock_boto_client):
        """Test that conversation history is included in requests"""
        mock_response_body = {
            "content": [{"text": "Response with context."}],
            "stop_reason": "end_turn"
        }

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(mock_response_body).encode()

        mock_client = Mock()
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client

        generator = AIGenerator("us-east-1", "test-model")
        history = "User: Previous question\nAssistant: Previous answer"

        generator.generate_response("Follow-up question", conversation_history=history)

        # Check that system prompt includes history
        call_args = mock_client.invoke_model.call_args
        request_body = json.loads(call_args[1]['body'])
        assert history in request_body['system']

    @patch('boto3.client')
    def test_bedrock_error_handling(self, mock_boto_client):
        """Test handling of Bedrock API errors"""
        from botocore.exceptions import ClientError

        mock_client = Mock()
        mock_client.invoke_model.side_effect = ClientError(
            error_response={'Error': {'Code': 'ValidationException'}},
            operation_name='InvokeModel'
        )
        mock_boto_client.return_value = mock_client

        generator = AIGenerator("us-east-1", "test-model")
        result = generator.generate_response("Test query")

        assert "Sorry, I encountered an error" in result

    @patch('boto3.client')
    def test_tool_choice_configuration(self, mock_boto_client):
        """Test that tool_choice is configured when tools are provided"""
        mock_response_body = {
            "content": [{"text": "Response"}],
            "stop_reason": "end_turn"
        }

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(mock_response_body).encode()

        mock_client = Mock()
        mock_client.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_client

        generator = AIGenerator("us-east-1", "test-model")
        tools = [{"name": "search_course_content"}]

        generator.generate_response("Test", tools=tools)

        call_args = mock_client.invoke_model.call_args
        request_body = json.loads(call_args[1]['body'])
        assert "tools" in request_body
        assert request_body["tool_choice"] == {"type": "auto"}