"""port of https://github.com/The-Pocket/PocketFlow-Tutorial-Youtube-Made-Simple"""

import asyncio
import os
import yaml
from pydantic import BaseModel

from pocket_joe import (
    Message,
    policy,
    BaseContext,
    InMemoryRunner,
    MessageBuilder,
    TextPart,
    OptionSchema,
)
from examples.utils import CompletionsAdapter, transcribe_youtube_policy


# --- Pydantic Models ---
class TopicData(BaseModel):
    """Topic with associated questions."""
    title: str
    questions: list[str]


class ExtractTopicsResult(BaseModel):
    """Result of topic extraction."""
    topics: list[TopicData]


class ProcessedQuestion(BaseModel):
    """Rephrased question with answer."""
    original: str
    rephrased: str
    answer: str


class ProcessTopicResult(BaseModel):
    """Result of topic processing."""
    rephrased_title: str
    questions: list[ProcessedQuestion]


# --- LLM Policy ---
@policy.tool(description="OpenAI-compatible chat completions")
async def llm_policy(
    observations: list[Message],
    options: list[OptionSchema] | None = None,
) -> list[Message]:
    """Call chat completions API and return option_call or text messages.

    Args:
        observations: Conversation history as pocket-joe Messages.
        options: Optional list of OptionSchema for tool/function calling.

    Returns:
        List of Messages containing text and/or option_call payloads.
    """
    adapter = CompletionsAdapter(observations, options)
    client = CompletionsAdapter.client(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    response = await client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=adapter.messages,  # type: ignore[arg-type]
        tools=adapter.tools or [],  # type: ignore[arg-type]
    )
    return adapter.decode(response, policy="llm_policy")


# --- Policies ---
@policy.tool(description="Extract interesting topics and questions from YouTube transcript")
async def extract_topics_policy(
    title: str,
    transcript: str,
) -> ExtractTopicsResult:
    """Extract interesting topics and generate questions from transcript.

    Args:
        title: Video title.
        transcript: Video transcript text.

    Returns:
        ExtractTopicsResult containing up to 5 topics with questions.
    """
    prompt = f"""
You are an expert content analyzer. Given a YouTube video transcript, identify at most 5 most interesting topics discussed and generate at most 3 most thought-provoking questions for each topic.

VIDEO TITLE: {title}

TRANSCRIPT:
{transcript}

IMPORTANT: Return ONLY valid YAML with this EXACT structure. Do not add any text before or after the YAML block.

```yaml
topics:
  - title: "First Topic Title"
    questions:
      - "Question 1 about first topic?"
      - "Question 2 about first topic?"
  - title: "Second Topic Title"
    questions:
      - "Question 1 about second topic?"
```
"""

    system_builder = MessageBuilder(policy="system", role_hint_for_llm="system")
    system_builder.add_text("You are a content analysis assistant.")
    system_message = system_builder.to_message()

    prompt_builder = MessageBuilder(policy="user", role_hint_for_llm="user")
    prompt_builder.add_text(prompt)
    prompt_message = prompt_builder.to_message()

    ctx = AppContext.get_ctx()
    history = [system_message, prompt_message]

    response = await ctx.llm(observations=history, options=[])

    # Extract YAML from response
    last_msg = next((msg for msg in reversed(response) if msg.parts), None)
    if last_msg and last_msg.parts:
        text_parts = [p for p in last_msg.parts if isinstance(p, TextPart)]
        content = text_parts[0].text if text_parts else ""
    else:
        content = ""

    yaml_content = content.split("```yaml")[1].split("```")[0].strip() if "```yaml" in content else content

    try:
        parsed = yaml.safe_load(yaml_content)
        return ExtractTopicsResult.model_validate(parsed)
    except (yaml.YAMLError, Exception) as e:
        print(f"YAML parsing error in extract_topics: {e}")
        print(f"Received content:\n{yaml_content}")
        return ExtractTopicsResult(topics=[])

@policy.tool(description="Rephrase topics and questions, generate simple answers")
async def process_topic_policy(
    topic_title: str,
    questions: list[str],
    transcript: str,
) -> ProcessTopicResult:
    """Rephrase topic title and questions, generate simple answers.

    Args:
        topic_title: Original topic title.
        questions: List of questions about the topic.
        transcript: Video transcript for context.

    Returns:
        ProcessTopicResult with rephrased title and Q&A pairs.
    """
    prompt = f"""You are a content analyst. Given a topic and questions from a YouTube video, rephrase them to be clear and concise, then provide accurate, informative answers.

TOPIC: {topic_title}

QUESTIONS:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions)])}

FULL TRANSCRIPT:
{transcript}

Instructions:
1. Rephrase the topic title to be clear and engaging (max 10 words)
2. Rephrase each question to be direct and specific (max 20 words)
3. Answer each question:
- Use markdown formatting (**bold** for emphasis, *italic* for technical terms)
- Use bullet points or numbered lists where appropriate
- Be concise but informative (2-3 sentences or 80-120 words)
- Base answers strictly on the transcript content
- Avoid condescending language

IMPORTANT: Return ONLY valid YAML with this EXACT structure. Use quoted strings, not pipe blocks.

```yaml
rephrased_title: "Clear, engaging topic title"
questions:
  - original: "First question here"
    rephrased: "Rephrased first question"
    answer: "Answer based on transcript"
  - original: "Second question here"
    rephrased: "Rephrased second question"
    answer: "Answer based on transcript"
```
"""

    system_builder = MessageBuilder(policy="system", role_hint_for_llm="system")
    system_builder.add_text("You are a content simplification assistant.")
    system_message = system_builder.to_message()

    prompt_builder = MessageBuilder(policy="user", role_hint_for_llm="user")
    prompt_builder.add_text(prompt)
    prompt_message = prompt_builder.to_message()

    history = [system_message, prompt_message]
    ctx = AppContext.get_ctx()

    response = await ctx.llm(observations=history, options=[])

    # Extract YAML from response
    last_msg = next((msg for msg in reversed(response) if msg.parts), None)
    if last_msg and last_msg.parts:
        text_parts = [p for p in last_msg.parts if isinstance(p, TextPart)]
        content = text_parts[0].text if text_parts else ""
    else:
        content = ""

    yaml_content = content.split("```yaml")[1].split("```")[0].strip() if "```yaml" in content else content

    try:
        parsed = yaml.safe_load(yaml_content)
        return ProcessTopicResult.model_validate(parsed)
    except (yaml.YAMLError, Exception) as e:
        print(f"YAML parsing error in process_topic: {e}")
        print(f"Received content:\n{yaml_content}")
        return ProcessTopicResult(rephrased_title=topic_title, questions=[])

@policy.tool(description="Process YouTube video to extract topics, questions, and generate ELI5 answers")
async def youtube_summarizer(
    url: str,
) -> list[Message]:
    """Process YouTube video to extract topics, questions, and generate ELI5 answers.

    Args:
        url: YouTube video URL.

    Returns:
        List of Messages with formatted markdown summary.
    """
    print(f"\n--- Processing YouTube URL: {url} ---")
    
    ctx = AppContext.get_ctx()

    # Step 1: Get video info
    video_info = await ctx.transcribe_youtube(url=url)

    if video_info.error:
        builder = MessageBuilder(policy="youtube_summarizer")
        builder.add_text(f"Error: {video_info.error}")
        return [builder.to_message()]

    print(f"Video: {video_info.title}")
    print(f"Transcript length: {len(video_info.transcript)} chars")

    # Step 2: Extract topics and questions
    print("\n--- Extracting topics and questions ---")
    topics_data = await ctx.extract_topics(
        title=video_info.title,
        transcript=video_info.transcript
    )
    topics = topics_data.topics
    print(f"Found {len(topics)} topics")

    # Step 3: Process each topic concurrently
    print("\n--- Processing topics ---")

    # Create tasks for all topics with questions
    tasks = []
    topic_indices = []
    for i, topic in enumerate(topics):
        if not topic.questions:
            continue

        print(f"Queuing topic {i+1}/{len(topics)}: {topic.title}")
        tasks.append(ctx.process_topic(
            topic_title=topic.title,
            questions=topic.questions,
            transcript=video_info.transcript
        ))
        topic_indices.append(i)
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Build processed topics list
    processed_topics = []
    for i, processed in zip(topic_indices, results):
        processed_topics.append({
            "original_title": topics[i].title,
            "rephrased_title": processed.rephrased_title,
            "questions": processed.questions
        })

    # Step 4: Format output
    print("\n--- Generating Summary ---")
    output = f"""
# {video_info.title}

**Video ID**: {video_info.video_id}
**Thumbnail**: {video_info.thumbnail_url}

---

"""
    for topic in processed_topics:
        output += f"## {topic['rephrased_title']}\n\n"
        for q in topic['questions']:
            output += f"### {q.rephrased}\n\n"
            output += f"{q.answer}\n\n"
    
    print("\n" + "=" * 50)
    print("Processing completed successfully!")
    print("=" * 50 + "\n")
    
    builder = MessageBuilder(policy="youtube_summarizer")
    builder.add_text(output)
    return [builder.to_message()]


# --- App Context ---
class AppContext(BaseContext):
    def __init__(self, runner):
        super().__init__(runner)
        self.llm = self._bind(llm_policy)
        self.transcribe_youtube = self._bind(transcribe_youtube_policy)
        self.extract_topics = self._bind(extract_topics_policy)
        self.process_topic = self._bind(process_topic_policy)
        self.youtube_summarizer = self._bind(youtube_summarizer)


# --- Main Execution ---
async def main():
    print("--- Starting YouTube Summarizer ---")
    
    # TODO: Replace with your YouTube URL
    # url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    url = "https://youtu.be/h_Zk4fDDcSY?si=LaxkHlRgWTCzq1n5"
    
    runner = InMemoryRunner()
    ctx = AppContext(runner)
    result = await ctx.youtube_summarizer(url=url)
    
    # Print summary
    output_text = str(result[-1])
    print("\n" + output_text)

    # Optionally save to file
    if output_text:
        with open("youtube_summary.md", "w") as f:
            f.write(output_text)
        print("\nSummary saved to youtube_summary.md")
    
    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
