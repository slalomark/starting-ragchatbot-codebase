import boto3
import json
from typing import List, Optional, Dict, Any
from botocore.exceptions import ClientError

class AIGenerator:
    """Handles interactions with AWS Bedrock Claude models for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, aws_region: str, model_id: str):
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=aws_region
        )
        self.model_id = model_id

        # Pre-build base inference parameters
        self.base_params = {
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare messages for Bedrock format
        messages = [{"role": "user", "content": query}]

        # Prepare the request body for Bedrock
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_content,
            "messages": messages,
            **self.base_params
        }

        # Add tools if available
        if tools:
            request_body["tools"] = tools
            request_body["tool_choice"] = {"type": "auto"}

        try:
            # Make the request to Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json'
            )

            # Parse the response
            response_body = json.loads(response['body'].read())

            # Handle tool execution if needed
            if response_body.get("stop_reason") == "tool_use" and tool_manager:
                return self._handle_tool_execution(response_body, request_body, tool_manager)

            # Return direct response
            return response_body["content"][0]["text"]

        except ClientError as e:
            print(f"Error calling Bedrock: {e}")
            return "Sorry, I encountered an error processing your request. Please try again."

    def _handle_tool_execution(self, initial_response: Dict[str, Any], base_request: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_request: Base request parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_request["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response["content"]})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response["content"]:
            if content_block.get("type") == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block["name"],
                    **content_block["input"]
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block["id"],
                    "content": tool_result
                })

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final request without tools
        final_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": base_request["system"],
            "messages": messages,
            **self.base_params
        }

        try:
            # Get final response
            final_response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(final_request),
                contentType='application/json'
            )

            final_response_body = json.loads(final_response['body'].read())
            return final_response_body["content"][0]["text"]

        except ClientError as e:
            print(f"Error calling Bedrock for final response: {e}")
            return "Sorry, I encountered an error processing the tool results. Please try again."