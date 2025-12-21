"""
Parallel image generation and editing example using pocket-joe.

This demonstrates:
1. Generating images from the same prompt using BOTH providers IN PARALLEL
2. Editing both generated images with a follow-up prompt IN PARALLEL
3. Using asyncio.gather for concurrent API calls
4. Comparing outputs from different image models

Uses:
- Gemini 2.5 Flash Image (google-genai)
- OpenAI gpt-image-1.5 (openai)

Requirements:
    pip install google-genai openai python-dotenv

Environment:
    GOOGLE_API_KEY - Your Google AI API key (can be in .env file)
    OPENAI_API_KEY - Your OpenAI API key (can be in .env file)
"""

import asyncio
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
from utils.openai_image_adapter import OpenAIImageAdapter

# Load environment variables from .env file
load_dotenv()

# Models
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
OPENAI_IMAGE_MODEL = "gpt-image-1.5"

# Output directory for generated images
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Gemini Policies ---

@policy.tool(description="Generate an image from a text prompt using Gemini")
async def gemini_image_gen(prompt: str) -> list[Message]:
    """Generate an image from a text prompt using Gemini.

    Args:
        prompt: Text description of the image to generate.

    Returns:
        List of Messages with generation result (text response + image data).
    """
    client = GeminiAdapter.client()
    
    input_msg = MessageBuilder(policy="user", role_hint_for_llm="user")
    input_msg.add_text(prompt)
    
    adapter = GeminiAdapter([input_msg.to_message()])

    response = await client.aio.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=adapter.contents,  # type: ignore
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    return adapter.decode(response, policy="gemini")


@policy.tool(description="Edit an existing image using Gemini")
async def gemini_image_edit(source_image_path: str, edit_prompt: str) -> list[Message]:
    """Edit an existing image based on a text prompt using Gemini.

    Args:
        source_image_path: Path to the source image to edit.
        edit_prompt: Instructions for how to edit the image.

    Returns:
        List of Messages with edit result (text response + image data).
    """
    client = GeminiAdapter.client()
    
    input_msg = MessageBuilder(policy="user", role_hint_for_llm="user")
    input_msg.add_text(edit_prompt)
    input_msg.add_image_path(source_image_path)
    
    adapter = GeminiAdapter([input_msg.to_message()])

    response = await client.aio.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=adapter.contents,  # type: ignore
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    return adapter.decode(response, policy="gemini")


# --- OpenAI Policies ---

@policy.tool(description="Generate an image from a text prompt using OpenAI")
async def openai_image_gen(prompt: str) -> list[Message]:
    """Generate an image from a text prompt using OpenAI.

    Args:
        prompt: Text description of the image to generate.

    Returns:
        List of Messages with generation result.
    """
    client = OpenAIImageAdapter.client()
    adapter = OpenAIImageAdapter()

    response = await client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    return adapter.decode(response, policy="openai")


@policy.tool(description="Edit an existing image using OpenAI")
async def openai_image_edit(source_image_path: str, edit_prompt: str) -> list[Message]:
    """Edit an existing image based on a text prompt using OpenAI.

    Args:
        source_image_path: Path to the source image to edit.
        edit_prompt: Instructions for how to edit the image.

    Returns:
        List of Messages with edit result.
    """
    client = OpenAIImageAdapter.client()
    adapter = OpenAIImageAdapter()

    # Build input message with image
    input_msg = MessageBuilder(policy="user", role_hint_for_llm="user")
    input_msg.add_text(edit_prompt)
    input_msg.add_image_path(source_image_path)
    messages = [input_msg.to_message()]

    # Extract prompt and images using adapter
    prompt = adapter.encode_prompt(messages)
    images = adapter.encode_images(messages)

    # OpenAI edit requires a file tuple (filename, bytes, mime) not raw bytes
    if images:
        filename, data = images[0]
        image_file = (filename, data, "image/png")
    else:
        data = Path(source_image_path).read_bytes()
        image_file = ("image.png", data, "image/png")

    response = await client.images.edit(
        model=OPENAI_IMAGE_MODEL,
        image=image_file,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    return adapter.decode(response, policy="openai")


# --- Orchestrator Policy ---

@policy.tool(description="Run parallel image generation and editing demo")
async def parallel_image_demo(gen_prompt: str, edit_prompt: str) -> dict[str, list[Message]]:
    """Orchestrate parallel image generation and editing with both providers.

    Args:
        gen_prompt: Prompt for initial image generation.
        edit_prompt: Prompt for editing the generated image.

    Returns:
        Dict mapping provider name to message history.
    """
    ctx = AppContext.get_ctx()
    results: dict[str, list[Message]] = {"gemini": [], "openai": []}

    # --- Step 1: Generate initial images IN PARALLEL ---
    print("\n--- Step 1: Generating initial images (parallel) ---")
    print("  • Gemini (gemini-2.5-flash-image)")
    print("  • OpenAI (gpt-image-1.5)")

    gemini_gen, openai_gen = await asyncio.gather(
        ctx.gemini_image_gen(prompt=gen_prompt),
        ctx.openai_image_gen(prompt=gen_prompt),
    )

    results["gemini"].extend(gemini_gen)
    results["openai"].extend(openai_gen)

    # Save generated images
    for provider, messages in [("gemini", gemini_gen), ("openai", openai_gen)]:
        image_part = next((p for p in iter_parts(messages, MediaPart) if p.data_b64), None)
        if image_part:
            path = OUTPUT_DIR / f"01_generated_{provider}.png"
            path.write_bytes(image_part.get_bytes())
            print(f"  ✓ {provider}: saved {path.name}")
        else:
            print(f"  ✗ {provider}: no image generated")

    # --- Step 2: Edit both images IN PARALLEL ---
    print("\n--- Step 2: Editing images (parallel) ---")

    gemini_path = str(OUTPUT_DIR / "01_generated_gemini.png")
    openai_path = str(OUTPUT_DIR / "01_generated_openai.png")

    # Only edit if both images exist
    gemini_exists = Path(gemini_path).exists()
    openai_exists = Path(openai_path).exists()

    edit_tasks = []
    edit_providers = []

    if gemini_exists:
        edit_tasks.append(ctx.gemini_image_edit(source_image_path=gemini_path, edit_prompt=edit_prompt))
        edit_providers.append("gemini")
    if openai_exists:
        edit_tasks.append(ctx.openai_image_edit(source_image_path=openai_path, edit_prompt=edit_prompt))
        edit_providers.append("openai")

    if edit_tasks:
        edit_results = await asyncio.gather(*edit_tasks)

        for provider, edit_msgs in zip(edit_providers, edit_results):
            results[provider].extend(edit_msgs)
            image_part = next((p for p in iter_parts(edit_msgs, MediaPart) if p.data_b64), None)
            if image_part:
                path = OUTPUT_DIR / f"02_edited_{provider}.png"
                path.write_bytes(image_part.get_bytes())
                print(f"  ✓ {provider}: saved {path.name}")

    return results


# --- App Context ---

class AppContext(BaseContext):
    """Context for parallel image generation demo."""

    def __init__(self, runner: Any):
        super().__init__(runner)
        self.gemini_image_gen = self._bind(gemini_image_gen)
        self.gemini_image_edit = self._bind(gemini_image_edit)
        self.openai_image_gen = self._bind(openai_image_gen)
        self.openai_image_edit = self._bind(openai_image_edit)
        self.parallel_image_demo = self._bind(parallel_image_demo)


# --- Main ---

async def main():
    """Run the parallel image generation and editing demo."""
    print("=" * 60)
    print("pocket-joe Parallel Image Generation & Editing Demo")
    print("=" * 60)
    print("\nRunning Gemini and OpenAI image models IN PARALLEL")

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

    import time
    start = time.perf_counter()

    results = await ctx.parallel_image_demo(
        gen_prompt=gen_prompt,
        edit_prompt=edit_prompt,
    )

    elapsed = time.perf_counter() - start

    # Print text responses
    print("\n--- Text Responses ---")
    for provider, messages in results.items():
        texts = [p.text for p in iter_parts(messages, TextPart)]
        if texts:
            print(f"\n{provider.upper()}:")
            for text in texts:
                truncated = text[:150] + "..." if len(text) > 150 else text
                print(f"  {truncated}")

    print("\n" + "=" * 60)
    print(f"Completed in {elapsed:.1f}s (both providers ran in parallel)")
    print(f"Check {OUTPUT_DIR} for generated images:")
    print("  • 01_generated_gemini.png / 01_generated_openai.png")
    print("  • 02_edited_gemini.png / 02_edited_openai.png")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
