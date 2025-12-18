"""
Image generation and editing example using pocket-joe.

This demonstrates:
1. Generating an image from a text prompt
2. Editing the generated image with a follow-up prompt
3. Using policies for different image generation backends

Currently uses Gemini 2.5 Flash for image gen/edit.
Architecture supports swapping in other models (DALL-E, Stable Diffusion, etc.)

Requirements:
    pip install google-genai pillow python-dotenv

Environment:
    GOOGLE_API_KEY - Your Google AI API key (can be in .env file)
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.genai import types

from pocket_joe import (
    Message,
    policy,
    BaseContext,
    InMemoryRunner,
    MessageBuilder,
    TextPart,
    MediaPart,
    iter_parts,
)
from utils.gemini_adapter import GeminiAdapter

# Load environment variables from .env file
load_dotenv()

# Model for image generation/editing
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"


# Output directory for generated images
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Policies ---

@policy.tool(description="Generate an image from a text prompt using Gemini")
async def gemini_image_gen(
    prompt: str,
    output_filename: str = "generated.png",
) -> list[Message]:
    """Generate an image from a text prompt.

    Args:
        prompt: Text description of the image to generate.
        output_filename: Filename for the output image.

    Returns:
        List of Messages with generation result (text response + image data).
    """
    client = GeminiAdapter.client()
    
    # Build input message
    input_msg = MessageBuilder(policy="user", role_hint_for_llm="user")
    input_msg.add_text(prompt)
    
    adapter = GeminiAdapter([input_msg.to_message()])

    response = await client.aio.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=adapter.contents, # type: ignore
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # Decode response to Messages
    messages = adapter.decode(response, policy="gemini_image_gen")
    
    # Save first image to disk
    output_path = OUTPUT_DIR / output_filename
    image_part = next((p for p in iter_parts(messages, MediaPart) if p.data_b64), None)
    if image_part:
        output_path.write_bytes(image_part.get_bytes())

    return messages


@policy.tool(description="Edit an existing image using Gemini")
async def gemini_image_edit(
    source_image_path: str,
    edit_prompt: str,
    output_filename: str = "edited.png",
) -> list[Message]:
    """Edit an existing image based on a text prompt.

    Args:
        source_image_path: Path to the source image to edit.
        edit_prompt: Instructions for how to edit the image.
        output_filename: Filename for the output image.

    Returns:
        List of Messages with edit result (text response + image data).
    """
    client = GeminiAdapter.client()
    
    # Build input message with text and image
    input_msg = MessageBuilder(policy="user", role_hint_for_llm="user")
    input_msg.add_text(edit_prompt)
    input_msg.add_image_path(source_image_path)
    
    adapter = GeminiAdapter([input_msg.to_message()])

    response = await client.aio.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=adapter.contents, # type: ignore
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # Decode response to Messages
    messages = adapter.decode(response, policy="gemini_image_edit")
    
    # Save first image to disk
    output_path = OUTPUT_DIR / output_filename
    image_part = next((p for p in iter_parts(messages, MediaPart) if p.data_b64), None)
    if image_part:
        output_path.write_bytes(image_part.get_bytes())

    return messages


@policy.tool(description="Run image generation and editing demo")
async def image_demo(
    gen_prompt: str,
    edit_prompt: str,
) -> list[Message]:
    """Orchestrate image generation followed by editing.

    Args:
        gen_prompt: Prompt for initial image generation.
        edit_prompt: Prompt for editing the generated image.

    Returns:
        List of Messages documenting the workflow.
    """
    ctx = AppContext.get_ctx()
    messages: list[Message] = []

    # Step 1: Generate initial image
    print("\n--- Step 1: Generating initial image ---")

    # Build user prompt message
    user_builder = MessageBuilder(policy="user", role_hint_for_llm="user")
    user_builder.add_text(gen_prompt)
    messages.append(user_builder.to_message())

    # Call generation policy (saves to 01_generated.png)
    gen_result = await ctx.gemini_image_gen(
        prompt=gen_prompt,
        output_filename="01_generated.png",
    )
    messages.extend(gen_result)

    # Print response and check if image was generated
    for part in iter_parts(gen_result, TextPart):
        print(part.text)
    
    has_image = any(p.data_b64 for p in iter_parts(gen_result, MediaPart))
    if has_image:
        print("[Image generated]")
    else:
        error_builder = MessageBuilder(policy="image_demo")
        error_builder.add_text("Error: No image was generated")
        return messages + [error_builder.to_message()]

    # The generated image was saved to disk by gemini_image_gen
    generated_image_path = str(OUTPUT_DIR / "01_generated.png")

    # Step 2: Edit the generated image
    print("\n--- Step 2: Editing image ---")

    # Build edit request message
    edit_builder = MessageBuilder(policy="user", role_hint_for_llm="user")
    edit_builder.add_text(edit_prompt)
    messages.append(edit_builder.to_message())

    # Call edit policy
    edit_result = await ctx.gemini_image_edit(
        source_image_path=generated_image_path,
        edit_prompt=edit_prompt,
        output_filename="02_edited.png",
    )
    messages.extend(edit_result)

    # Print edit response
    for part in iter_parts(edit_result, TextPart):
        print(part.text)
    if any(p.data_b64 for p in iter_parts(edit_result, MediaPart)):
        print("[Image edited]")

    return messages


# --- App Context ---

class AppContext(BaseContext):
    """Context for image generation demo."""

    def __init__(self, runner: Any):
        super().__init__(runner)
        self.gemini_image_gen = self._bind(gemini_image_gen)
        self.gemini_image_edit = self._bind(gemini_image_edit)
        self.image_demo = self._bind(image_demo)


# --- Main ---

async def main():
    """Run the image generation and editing demo."""
    print("=" * 50)
    print("pocket-joe Image Generation & Editing Demo")
    print("=" * 50)

    runner = InMemoryRunner()
    ctx = AppContext(runner)

    gen_prompt = """
    A cozy coffee shop interior with:
    - Warm lighting from pendant lamps
    - A wooden counter with an espresso machine
    - Plants on shelves
    - Morning sunlight streaming through large windows
    """

    edit_prompt = """
    Edit this image to:
    - Add a cat sleeping on one of the chairs
    - Change the time to evening with warm golden hour light
    - Add rain visible through the windows
    """

    result = await ctx.image_demo(
        gen_prompt=gen_prompt,
        edit_prompt=edit_prompt,
    )

    print("\n" + "=" * 50)
    print("Conversation history:")
    print("=" * 50)
    for i, msg in enumerate(result):
        print(f"\n[{i}] policy={msg.policy}, role={msg.role_hint_for_llm}")
        for part in iter_parts([msg], TextPart):
            text = part.text[:100] + "..." if len(part.text) > 100 else part.text
            print(f"    {text}")

    print("\n" + "=" * 50)
    print("Demo complete!")
    print(f"Check {OUTPUT_DIR} for generated images")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
