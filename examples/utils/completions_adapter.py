"""
Completions adapter for pocket-joe.
Encodes pocket-joe Messages to chat completions format, decodes responses back.
Works with OpenAI, Azure OpenAI, OpenRouter, and any OpenAI-compatible API.

Requirements: openai
Install with: pip install openai
"""
import json
from typing import Any
import uuid

from pocket_joe import Message, OptionSchema
from pocket_joe import (
    TextPart,
    OptionCallPayload,
    OptionResultPayload,
    MessageBuilder,
)

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError(
        "openai is required for this adapter. "
        "Install with: pip install openai"
    )


class CompletionsAdapter:
    """Adapter for chat completions API (OpenAI-compatible).
    
    Encodes pocket-joe Messages to chat completions format and decodes
    responses back to Messages. Does not abstract the API call.
    
    Works with any OpenAI-compatible API:
    - OpenAI
    - Azure OpenAI
    - OpenRouter
    - Local models (LM Studio, Ollama, etc.)
    
    Usage:
        client = CompletionsAdapter.client()
        adapter = CompletionsAdapter(observations, options)
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=adapter.messages,
            tools=adapter.tools or None,
            temperature=0.7,
        )
        
        messages = adapter.decode(response, policy="my_agent")
    """
    
    def __init__(
        self,
        observations: list[Message],
        options: list[OptionSchema] | None = None,
    ):
        """Initialize adapter with observations and options.
        
        Args:
            observations: List of Messages representing conversation history
            options: Optional list of OptionSchema for tool/function calling
        """
        self.messages = self._encode_messages(observations)
        self.tools = self._encode_tools(options)
    
    def decode(
        self,
        response: Any,
        policy: str = "completions",
    ) -> list[Message]:
        """Decode completions response to pocket-joe Messages.
        
        Args:
            response: ChatCompletion response from completions API
            policy: Policy name for the output messages
            
        Returns:
            List of Messages containing text and/or option_call messages
        """
        return self._decode_response(response, policy)
    
    @staticmethod
    def client(**kwargs) -> AsyncOpenAI:
        """Create an AsyncOpenAI client.
        
        Args:
            **kwargs: Passed to AsyncOpenAI constructor
                - api_key: API key (defaults to OPENAI_API_KEY env var)
                - base_url: API base URL for alternative providers
                - etc.
            
        Returns:
            AsyncOpenAI client instance
            
        Examples:
            # OpenAI (default)
            client = CompletionsAdapter.client()
            
            # OpenRouter
            client = CompletionsAdapter.client(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
            
            # Azure OpenAI
            from openai import AsyncAzureOpenAI
            client = AsyncAzureOpenAI(...)
        """
        return AsyncOpenAI(**kwargs)
    
    # --- Internal encoding/decoding ---
    
    def _encode_messages(self, in_msgs: list[Message]) -> list[dict[str, Any]]:
        """Convert pocket-joe Messages to chat completions format."""
        # Build mapping of invocation_id -> option_result
        tool_results: dict[str, Message] = {}
        for msg in in_msgs:
            if msg.payload and isinstance(msg.payload, OptionResultPayload):
                tool_results[msg.payload.invocation_id] = msg

        messages: list[dict[str, Any]] = []
        for msg in in_msgs:
            # Handle parts messages (text + media)
            if msg.parts:
                text_parts = [p for p in msg.parts if isinstance(p, TextPart)]
                content = " ".join(p.text for p in text_parts)
                role = msg.role_hint_for_llm or "assistant"
                messages.append({"role": role, "content": content})

            # Handle option_call messages
            elif msg.payload and isinstance(msg.payload, OptionCallPayload):
                call_payload = msg.payload
                invocation_id = call_payload.invocation_id

                # Only include if we have the corresponding result (complete pair)
                if invocation_id not in tool_results:
                    continue

                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "type": "function",
                        "id": invocation_id,
                        "function": {
                            "name": call_payload.option_name,
                            "arguments": json.dumps(call_payload.arguments)
                        }
                    }],
                })

                result_msg = tool_results[invocation_id]
                result_payload = result_msg.payload
                if isinstance(result_payload, OptionResultPayload):
                    result = result_payload.result
                    if isinstance(result, str):
                        content = result
                    else:
                        content = json.dumps(result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": invocation_id,
                        "content": content
                    })

        return messages
    
    def _encode_tools(self, options: list[OptionSchema] | None) -> list[dict[str, Any]] | None:
        """Convert OptionSchema list to chat completions tool format."""
        if not options:
            return None
        
        tools: list[dict[str, Any]] = []
        for option in options:
            tools.append({
                "type": "function",
                "function": option.model_dump()
            })
        return tools
    
    def _decode_response(self, response: Any, policy: str) -> list[Message]:
        """Convert completions response to pocket-joe Messages."""
        new_messages: list[Message] = []
        msg = response.choices[0].message

        if msg.content:
            builder = MessageBuilder(policy=policy, role_hint_for_llm="assistant")
            builder.add_text(msg.content)
            new_messages.append(builder.to_message())

        if msg.tool_calls:
            for tc in msg.tool_calls:
                new_messages.append(Message(
                    id=str(uuid.uuid4()),
                    policy=policy,
                    role_hint_for_llm="assistant",
                    payload=OptionCallPayload(
                        invocation_id=tc.id,
                        option_name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    )
                ))

        return new_messages
