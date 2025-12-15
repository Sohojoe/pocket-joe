"""Message models and builders for PocketJoe.

This module defines the core message types used throughout the framework:
- Part models (TextPart, MediaPart) for content
- Payload models (OptionCallPayload, OptionResultPayload) for actions
- Message model for the atomic communication unit
- Builder classes for ergonomic message construction
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, HttpUrl
import uuid


# ============================================================================
# Part Models (for messages with content)
# ============================================================================

class TextPart(BaseModel):
    """Text content part of a message."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["text"] = "text"
    text: str


class MediaPart(BaseModel):
    """Media content part of a message (image, audio, video, document).

    Media is URL-based and modality-aware. The modality is required and
    semantic, while MIME type is optional best-effort metadata.
    """
    model_config = ConfigDict(frozen=True)

    kind: Literal["media"] = "media"
    modality: Literal["image", "audio", "video", "document"]
    url: HttpUrl

    # Optional metadata
    mime: Optional[str] = None
    prompt_hint: Optional[str] = None  # Non-semantic label/description for prompts
    id: Optional[str] = None  # Internal correlation/replay aid


# Union type for all part types
Part = TextPart | MediaPart


# ============================================================================
# Option Payload Models
# ============================================================================

class OptionCallPayload(BaseModel):
    """Payload for option_call messages (policy-selected actions)."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["option_call"] = "option_call"
    invocation_id: str  # Unique per call, pairs call ↔ result
    option_name: str
    arguments: dict[str, Any]


class OptionResultPayload(BaseModel):
    """Payload for option_result messages (results of executing options)."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["option_result"] = "option_result"
    invocation_id: str  # Must match corresponding option_call
    option_name: str
    result: Any
    is_error: bool = False

    # Structured error info (used when is_error=True)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retryable: Optional[bool] = None


# Union type for all payload types
Payload = OptionCallPayload | OptionResultPayload


# ============================================================================
# Message Model
# ============================================================================

class Message(BaseModel):
    """Immutable message structure for agent communication.

    Messages are the atomic units flowing through the system. They can contain:
    - Natural language + media (parts)
    - Policy-selected actions (option_call)
    - Results of executing options (option_result)

    Framework semantics:
    - `policy`: Which policy created the message
    - `step_num`: Groups messages in the same logical timestep

    Provider semantics (adapter concerns only):
    - `role_hint_for_llm`: Hint for provider adapters (user/assistant/tool/system)
    """
    model_config = ConfigDict(frozen=True)

    # Core framework fields
    policy: str  # Which policy created this message

    # Message content (mutually exclusive with payload)
    parts: Optional[list[Part]] = None

    # Option call/result payload (mutually exclusive with parts)
    payload: Optional[Payload] = None

    # Step grouping
    step_num: Optional[int] = None

    # Provider adapter hint (not used by core framework)
    role_hint_for_llm: Optional[Literal["user", "assistant", "tool", "system"]] = None

    # Unique identifier (framework-generated)
    id: str = ""

    def model_post_init(self, __context):
        """Validate that exactly one of parts or payload is set."""
        if self.parts is not None and self.payload is not None:
            raise ValueError("Message cannot have both parts and payload")
        if self.parts is None and self.payload is None:
            raise ValueError("Message must have either parts or payload")

    def __str__(self) -> str:
        """Return text content of message if available, empty string otherwise."""
        if self.parts:
            text_parts = [p.text for p in self.parts if isinstance(p, TextPart)]
            return " ".join(text_parts)
        return ""


# ============================================================================
# Message Builders
# ============================================================================

class MessageBuilder:
    """Builder for constructing messages with parts (text + media).

    Provides ergonomic construction methods and maintains consistency
    for step_num, policy, and role hints.
    """

    def __init__(
        self,
        policy: str,
        step_num: Optional[int] = None,
        role_hint_for_llm: Optional[Literal["user", "assistant", "tool", "system"]] = None,
    ):
        self.policy = policy
        self.step_num = step_num
        self.role_hint_for_llm = role_hint_for_llm
        self._parts: list[Part] = []
        self._last_option_call: Optional[Message] = None

    @classmethod
    def next_step(
        cls,
        last_message: Message,
        policy: Optional[str] = None,
        role_hint_for_llm: Optional[Literal["user", "assistant", "tool", "system"]] = None,
    ) -> "MessageBuilder":
        """Create a builder for the next step (increments step_num).

        Args:
            last_message: Previous message to anchor from
            policy: Policy name (defaults to last_message.policy)
            role_hint_for_llm: Role hint (defaults to last_message.role_hint_for_llm)
        """
        next_step_num = (last_message.step_num or 0) + 1
        return cls(
            policy=policy or last_message.policy,
            step_num=next_step_num,
            role_hint_for_llm=role_hint_for_llm or last_message.role_hint_for_llm,
        )

    @classmethod
    def continue_step(
        cls,
        last_message: Message,
        policy: Optional[str] = None,
        role_hint_for_llm: Optional[Literal["user", "assistant", "tool", "system"]] = None,
    ) -> "MessageBuilder":
        """Create a builder that continues the current step (keeps step_num).

        Args:
            last_message: Previous message to anchor from
            policy: Policy name (defaults to last_message.policy)
            role_hint_for_llm: Role hint (defaults to last_message.role_hint_for_llm)
        """
        return cls(
            policy=policy or last_message.policy,
            step_num=last_message.step_num,
            role_hint_for_llm=role_hint_for_llm or last_message.role_hint_for_llm,
        )

    def add_text(self, text: str) -> "MessageBuilder":
        """Add a text part to the message."""
        self._parts.append(TextPart(text=text))
        return self

    def add_image(
        self,
        url: str | HttpUrl,
        prompt_hint: Optional[str] = None,
        mime: Optional[str] = None,
        media_id: Optional[str] = None,
    ) -> "MessageBuilder":
        """Add an image media part to the message."""
        self._parts.append(MediaPart(
            modality="image",
            url=url,  # type: ignore
            mime=mime,
            prompt_hint=prompt_hint,
            id=media_id,
        ))
        return self

    def add_audio(
        self,
        url: str | HttpUrl,
        prompt_hint: Optional[str] = None,
        mime: Optional[str] = None,
        media_id: Optional[str] = None,
    ) -> "MessageBuilder":
        """Add an audio media part to the message."""
        self._parts.append(MediaPart(
            modality="audio",
            url=url,  # type: ignore
            mime=mime,
            prompt_hint=prompt_hint,
            id=media_id,
        ))
        return self

    def add_video(
        self,
        url: str | HttpUrl,
        prompt_hint: Optional[str] = None,
        mime: Optional[str] = None,
        media_id: Optional[str] = None,
    ) -> "MessageBuilder":
        """Add a video media part to the message."""
        self._parts.append(MediaPart(
            modality="video",
            url=url,  # type: ignore
            mime=mime,
            prompt_hint=prompt_hint,
            id=media_id,
        ))
        return self

    def add_document(
        self,
        url: str | HttpUrl,
        prompt_hint: Optional[str] = None,
        mime: Optional[str] = None,
        media_id: Optional[str] = None,
    ) -> "MessageBuilder":
        """Add a document media part to the message."""
        self._parts.append(MediaPart(
            modality="document",
            url=url,  # type: ignore
            mime=mime,
            prompt_hint=prompt_hint,
            id=media_id,
        ))
        return self

    def add_option_call(
        self,
        option_name: str,
        arguments: dict[str, Any],
        invocation_id: Optional[str] = None,
    ) -> "MessageBuilder":
        """Add an option call and immediately emit it as a separate message.

        Note: This creates and stores an option_call message separately from
        the parts message being built. The option_call message is accessible
        via last_option_call property.

        Args:
            option_name: Name of the option to call
            arguments: Arguments for the option
            invocation_id: Optional invocation ID (auto-generated if not provided)
        """
        inv_id = invocation_id or str(uuid.uuid4())

        option_call_msg = Message(
            id=str(uuid.uuid4()),
            policy=self.policy,
            step_num=self.step_num,
            role_hint_for_llm=self.role_hint_for_llm,
            payload=OptionCallPayload(
                invocation_id=inv_id,
                option_name=option_name,
                arguments=arguments,
            ),
        )

        self._last_option_call = option_call_msg
        return self

    @property
    def last_option_call(self) -> Optional[Message]:
        """Get the last option call created by this builder."""
        return self._last_option_call

    def to_message(self) -> Message:
        """Build and return the final message with all parts."""
        if not self._parts:
            raise ValueError("Cannot build message with no parts. Use add_text() or add_*() methods first.")

        return Message(
            id=str(uuid.uuid4()),
            policy=self.policy,
            step_num=self.step_num,
            role_hint_for_llm=self.role_hint_for_llm,
            parts=self._parts.copy(),
        )

    def to_messages(self) -> list[Message]:
        """Build and return all messages (parts message + any option calls).

        Returns a list containing:
        1. The parts message (if any parts were added)
        2. Any option_call messages created via add_option_call()
        """
        messages = []

        # Add parts message if we have parts
        if self._parts:
            messages.append(self.to_message())

        # Add option call message if one was created
        if self._last_option_call:
            messages.append(self._last_option_call)

        return messages


class OptionResultBuilder:
    """Builder for constructing option_result messages in response to option_call.

    Automatically copies invocation_id, option_name, and step_num from the
    option_call message to ensure consistency.
    """

    def __init__(self, option_call: Message, policy: Optional[str] = None):
        """Create a builder for an option_result message.

        Args:
            option_call: The option_call message this result responds to
            policy: Policy name (defaults to the invoked option's name)

        Raises:
            ValueError: If option_call is not an option_call message
        """
        if option_call.payload is None or not isinstance(option_call.payload, OptionCallPayload):
            raise ValueError("option_call must be an option_call message")

        self._option_call = option_call
        self._call_payload = option_call.payload
        self._policy = policy or option_call.payload.option_name

    @classmethod
    def response_to(cls, option_call: Message, policy: Optional[str] = None) -> "OptionResultBuilder":
        """Create a builder for responding to an option_call.

        Args:
            option_call: The option_call message to respond to
            policy: Policy name (defaults to the invoked option's name)
        """
        return cls(option_call, policy)

    def success(self, result: Any) -> Message:
        """Create a successful option_result message.

        Args:
            result: The result value from executing the option
        """
        return Message(
            id=str(uuid.uuid4()),
            policy=self._policy,
            step_num=self._option_call.step_num,
            role_hint_for_llm="tool",
            payload=OptionResultPayload(
                invocation_id=self._call_payload.invocation_id,
                option_name=self._call_payload.option_name,
                result=result,
                is_error=False,
            ),
        )

    def error(
        self,
        error_type: str,
        error_message: str,
        retryable: bool = False,
    ) -> Message:
        """Create an error option_result message.

        Args:
            error_type: Type of error that occurred
            error_message: Human-readable error message
            retryable: Whether the error is retryable
        """
        return Message(
            id=str(uuid.uuid4()),
            policy=self._policy,
            step_num=self._option_call.step_num,
            role_hint_for_llm="tool",
            payload=OptionResultPayload(
                invocation_id=self._call_payload.invocation_id,
                option_name=self._call_payload.option_name,
                result=None,
                is_error=True,
                error_type=error_type,
                error_message=error_message,
                retryable=retryable,
            ),
        )
