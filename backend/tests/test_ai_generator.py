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

    @patch('boto3.client')
    def test_sequential_tool_calls_two_rounds(self, mock_boto_client, mock_tool_manager):
        """Test that Claude can make 2 sequential tool calls in separate API rounds"""
        # Mock responses for: tool_use -> tool_use -> final response
        first_tool_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_001",
                    "name": "search_course_content",
                    "input": {
                        "query": "Python course",
                        "course_name": "Python Programming"
                    }
                }
            ],
            "stop_reason": "tool_use"
        }

        second_tool_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_002",
                    "name": "search_course_content",
                    "input": {
                        "query": "variables lesson",
                        "lesson_number": 2
                    }
                }
            ],
            "stop_reason": "tool_use"
        }

        final_response = {
            "content": [{"text": "Based on both searches, variables are fundamental in Python programming."}],
            "stop_reason": "end_turn"
        }

        mock_client = Mock()
        mock_boto_client.return_value = mock_client

        # Setup 3 API responses: round 1 tool use, round 2 tool use, final response
        responses = [
            {"body": Mock()},  # First tool use
            {"body": Mock()},  # Second tool use
            {"body": Mock()}   # Final response
        ]
        responses[0]["body"].read.return_value = json.dumps(first_tool_response).encode()
        responses[1]["body"].read.return_value = json.dumps(second_tool_response).encode()
        responses[2]["body"].read.return_value = json.dumps(final_response).encode()

        mock_client.invoke_model.side_effect = responses

        # Setup tool manager for 2 executions
        mock_tool_manager.execute_tool.side_effect = [
            "First search results about Python course",
            "Second search results about variables lesson"
        ]

        generator = AIGenerator("us-east-1", "test-model")
        tools = [{"name": "search_course_content", "description": "Search courses"}]

        result = generator.generate_response(
            "Find information about variables in Python course",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify 3 API calls were made (2 tool rounds + 1 final)
        assert mock_client.invoke_model.call_count == 3

        # Verify 2 tool executions
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify final response
        assert result == "Based on both searches, variables are fundamental in Python programming."

    @patch('boto3.client')
    def test_termination_after_no_tool_use(self, mock_boto_client, mock_tool_manager):
        """Test that conversation terminates when Claude doesn't request tools"""
        # Mock response without tool use
        direct_response = {
            "content": [{"text": "This is a direct answer without needing tools."}],
            "stop_reason": "end_turn"
        }

        mock_client = Mock()
        mock_boto_client.return_value = mock_client

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(direct_response).encode()
        mock_client.invoke_model.return_value = mock_response

        generator = AIGenerator("us-east-1", "test-model")
        tools = [{"name": "search_course_content"}]

        result = generator.generate_response(
            "What is 2+2?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should make only 1 API call and no tool executions
        assert mock_client.invoke_model.call_count == 1
        assert mock_tool_manager.execute_tool.call_count == 0
        assert result == "This is a direct answer without needing tools."

    @patch('boto3.client')
    def test_termination_after_max_rounds(self, mock_boto_client, mock_tool_manager):
        """Test that conversation terminates after 2 rounds and makes final call without tools"""
        # Mock responses for: tool_use -> tool_use -> final response (without tools)
        tool_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "search_course_content",
                    "input": {"query": "test"}
                }
            ],
            "stop_reason": "tool_use"
        }

        final_response = {
            "content": [{"text": "Final answer after max rounds reached."}],
            "stop_reason": "end_turn"
        }

        mock_client = Mock()
        mock_boto_client.return_value = mock_client

        # Setup responses: 2 tool uses + 1 final call
        responses = [
            {"body": Mock()},  # Round 1 tool use
            {"body": Mock()},  # Round 2 tool use
            {"body": Mock()}   # Final call without tools
        ]
        responses[0]["body"].read.return_value = json.dumps(tool_response).encode()
        responses[1]["body"].read.return_value = json.dumps(tool_response).encode()
        responses[2]["body"].read.return_value = json.dumps(final_response).encode()

        mock_client.invoke_model.side_effect = responses
        mock_tool_manager.execute_tool.return_value = "Search results"

        generator = AIGenerator("us-east-1", "test-model")
        tools = [{"name": "search_course_content"}]

        result = generator.generate_response(
            "Complex query requiring multiple searches",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should make 3 API calls (2 tool rounds + 1 final without tools)
        assert mock_client.invoke_model.call_count == 3

        # Should execute tools 2 times
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify final call was made without tools
        final_call_args = mock_client.invoke_model.call_args_list[2]
        final_request_body = json.loads(final_call_args[1]['body'])
        assert "tools" not in final_request_body

        assert result == "Final answer after max rounds reached."

    @patch('boto3.client')
    def test_tool_error_handling_during_sequential_calls(self, mock_boto_client, mock_tool_manager):
        """Test graceful handling of tool execution errors during sequential calls"""
        tool_use_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_456",
                    "name": "search_course_content",
                    "input": {"query": "test"}
                }
            ],
            "stop_reason": "tool_use"
        }

        mock_client = Mock()
        mock_boto_client.return_value = mock_client

        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(tool_use_response).encode()
        mock_client.invoke_model.return_value = mock_response

        # Mock tool execution to raise an exception
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        generator = AIGenerator("us-east-1", "test-model")
        tools = [{"name": "search_course_content"}]

        result = generator.generate_response(
            "Query that will cause tool error",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should make 1 API call and 1 failed tool execution
        assert mock_client.invoke_model.call_count == 1
        assert mock_tool_manager.execute_tool.call_count == 1

        # Should return error message
        assert "Sorry, I encountered an error with the search tool" in result

    @patch('boto3.client')
    def test_message_accumulation_across_rounds(self, mock_boto_client, mock_tool_manager):
        """Test that messages accumulate correctly across multiple rounds"""
        # Mock two tool use responses
        first_tool_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_001",
                    "name": "search_course_content",
                    "input": {"query": "first search"}
                }
            ],
            "stop_reason": "tool_use"
        }

        second_tool_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_002",
                    "name": "search_course_content",
                    "input": {"query": "second search"}
                }
            ],
            "stop_reason": "tool_use"
        }

        final_response = {
            "content": [{"text": "Final response"}],
            "stop_reason": "end_turn"
        }

        mock_client = Mock()
        mock_boto_client.return_value = mock_client

        responses = [
            {"body": Mock()},
            {"body": Mock()},
            {"body": Mock()}
        ]
        responses[0]["body"].read.return_value = json.dumps(first_tool_response).encode()
        responses[1]["body"].read.return_value = json.dumps(second_tool_response).encode()
        responses[2]["body"].read.return_value = json.dumps(final_response).encode()

        mock_client.invoke_model.side_effect = responses
        mock_tool_manager.execute_tool.side_effect = ["First result", "Second result"]

        generator = AIGenerator("us-east-1", "test-model")
        tools = [{"name": "search_course_content"}]

        generator.generate_response(
            "Test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify message structure in final call
        final_call_args = mock_client.invoke_model.call_args_list[2]
        final_request_body = json.loads(final_call_args[1]['body'])
        messages = final_request_body['messages']

        # Should have: user query + assistant response + tool results + assistant response + tool results
        assert len(messages) == 5
        assert messages[0]["role"] == "user"  # Original query
        assert messages[1]["role"] == "assistant"  # First tool use
        assert messages[2]["role"] == "user"  # First tool results
        assert messages[3]["role"] == "assistant"  # Second tool use
        assert messages[4]["role"] == "user"  # Second tool results