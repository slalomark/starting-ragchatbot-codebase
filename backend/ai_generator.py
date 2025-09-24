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
- **You can make multiple targeted searches (up to 2 rounds total)** for complex questions requiring:
  - Information from different courses or lessons
  - Comparisons between topics or courses
  - Multi-part questions needing separate searches
- Each search should be targeted and build upon previous results
- Synthesize all search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Complex questions**: Use multiple searches to gather comprehensive information, then provide complete synthesis
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
        Generate AI response with sequential tool calling support (up to 2 rounds).

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        messages = [{"role": "user", "content": query}]
        max_rounds = 2

        # Loop through up to 2 rounds of tool calling
        for round_num in range(max_rounds):
            # Make API call for this round
            response = self._make_api_call(messages, tools, conversation_history)

            if response is None:
                return "Sorry, I encountered an error processing your request. Please try again."

            # If no tool use or no tool manager, return response directly
            if response.get("stop_reason") != "tool_use" or not tool_manager:
                return self._extract_text_response(response)

            # Execute tools and prepare for next round
            messages = self._process_tool_round(messages, response, tool_manager)
            if messages is None:  # Tool execution failed
                return "Sorry, I encountered an error with the search tool. Please try again."

        # After max rounds, make final call without tools
        final_response = self._make_api_call(messages, tools=None, conversation_history=conversation_history)
        if final_response is None:
            return "Sorry, I encountered an error processing your request. Please try again."

        return self._extract_text_response(final_response)


    def _make_api_call(self, messages: List[Dict], tools: Optional[List] = None, conversation_history: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Make a single API call to Bedrock.

        Args:
            messages: List of conversation messages
            tools: Available tools for this call
            conversation_history: Previous conversation context

        Returns:
            Response body from Bedrock API or None on error
        """
        # Build system content efficiently
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

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

            # Parse and return the response
            return json.loads(response['body'].read())

        except ClientError as e:
            print(f"Error calling Bedrock: {e}")
            return None

    def _process_tool_round(self, messages: List[Dict], response: Dict[str, Any], tool_manager) -> Optional[List[Dict]]:
        """
        Execute tools and update message history for next round.

        Args:
            messages: Current message history
            response: API response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            Updated messages list or None on error
        """
        # Add AI's tool use response
        updated_messages = messages.copy()
        updated_messages.append({"role": "assistant", "content": response["content"]})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response["content"]:
            if content_block.get("type") == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block["name"],
                        **content_block["input"]
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block["id"],
                        "content": tool_result
                    })
                except Exception as e:
                    print(f"Error executing tool {content_block['name']}: {e}")
                    return None

        # Add tool results as single message
        if tool_results:
            updated_messages.append({"role": "user", "content": tool_results})

        return updated_messages

    def _extract_text_response(self, response: Dict[str, Any]) -> str:
        """
        Extract text response from Bedrock API response.

        Args:
            response: API response body

        Returns:
            Text content from response
        """
        if not response or "content" not in response:
            return "Sorry, I encountered an error processing your request. Please try again."

        content = response["content"]
        if content and isinstance(content, list) and len(content) > 0:
            return content[0].get("text", "Sorry, I couldn't generate a response.")

        return "Sorry, I couldn't generate a response."