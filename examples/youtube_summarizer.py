"""port of https://github.com/The-Pocket/PocketFlow-Tutorial-Youtube-Made-Simple"""

import asyncio
import yaml
from pocket_joe import (
    Message,
    policy, 
    BaseContext, 
    InMemoryRunner,
)
from examples.utils import openai_llm_policy_v1, transcribe_youtube_policy


# --- Policies ---
@policy.tool(description="Extract interesting topics and questions from YouTube transcript")
async def extract_topics_policy(
    title: str,
    transcript: str,
) -> list[Message]:
    """Extract interesting topics and generate questions from transcript.
    
    Args:
        title: Video title
        transcript: Video transcript

    Returns:
        List containing a single Message with payload containing topics array
    """
    prompt = f"""
You are an expert content analyzer. Given a YouTube video transcript, identify at most 5 most interesting topics discussed and generate at most 3 most thought-provoking questions for each topic.

VIDEO TITLE: {title}

TRANSCRIPT:
{transcript}

Format your response in YAML:

```yaml
topics:
- title: |
    First Topic Title
questions:
    - |
    Question 1 about first topic?
    - |
    Question 2 about first topic?
- title: |
    Second Topic Title
questions:
    - |
    Question 1 about second topic?
```
"""
    
    system_message = Message(
        actor="system",
        type="text",
        payload={"content": "You are a content analysis assistant."}
    )
    prompt_message = Message(
        actor="user",
        type="text",
        payload={"content": prompt}
    )

    ctx = AppContext.get_ctx()
    history = [system_message, prompt_message]

    response = await ctx.llm(observations=history, options=[])
    
    # Extract YAML from response
    content = response[-1].payload.get("content", "")
    yaml_content = content.split("```yaml")[1].split("```")[0].strip() if "```yaml" in content else content
    parsed = yaml.safe_load(yaml_content)
    
    result_message = Message(
        actor="extract_topics",
        type="action_result",
        payload={"topics": parsed.get("topics", [])[:5]}
    )
    
    return [result_message]

@policy.tool(description="Rephrase topics and questions, generate simple answers")
async def process_topic_policy(
    topic_title: str,
    questions: list[str],
    transcript: str,
) -> list[Message]:
    """
    Rephrase topic title and questions, generate simple answers.
    
    Args:
        topic_title: Original topic title
        questions: List of questions about the topic
        transcript: Video transcript for context

    Returns:
        List containing a single Message with rephrased topic and Q&A pairs
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

Format your response in YAML (ensure all questions are included):

```yaml
rephrased_title: |
Clear, engaging topic title
questions:
- original: |
    First question here
rephrased: |
    Rephrased first question
answer: |
    Answer based on transcript
- original: |
    Second question here
rephrased: |
    Rephrased second question
answer: |
    Answer based on transcript
```
"""
    
    system_message = Message(
        actor="system",
        type="text",
        payload={"content": "You are a content simplification assistant."}
    )
    prompt_message = Message(
        actor="user",
        type="text",
        payload={"content": prompt}
    )
    
    history = [system_message, prompt_message]
    ctx = AppContext.get_ctx()

    response = await ctx.llm(observations=history, options=[])
    
    # Extract YAML from response
    content = response[-1].payload.get("content", "")
    yaml_content = content.split("```yaml")[1].split("```")[0].strip() if "```yaml" in content else content
    parsed = yaml.safe_load(yaml_content)
    
    result_message = Message(
        actor="process_topic",
        type="action_result",
        payload={
            "rephrased_title": parsed.get("rephrased_title", topic_title),
            "questions": parsed.get("questions", [])
        }
    )
    
    return [result_message]

@policy.tool(description="Process YouTube video to extract topics, questions, and generate ELI5 answers")
async def youtube_summarizer(
    url: str,
) -> list[Message]:
    """
    Process YouTube video to extract topics, questions, and generate ELI5 answers.
    
    Args:
        url: YouTube video URL

    Returns:
        List containing Messages with video metadata and processed topics with Q&A
    """
    print(f"\n--- Processing YouTube URL: {url} ---")
    
    ctx = AppContext.get_ctx()

    # Step 1: Get video info
    result = await ctx.transcribe_youtube(url=url)
    video_info = result[0].payload
    
    if "error" in video_info:
        return [Message(
            actor="youtube_summarizer",
            type="text",
            payload={"content": f"Error: {video_info['error']}"}
        )]
    
    print(f"Video: {video_info['title']}")
    print(f"Transcript length: {len(video_info['transcript'])} chars")
    
    # Step 2: Extract topics and questions
    print("\n--- Extracting topics and questions ---")
    extract_result = await ctx.extract_topics(
        title=video_info["title"],
        transcript=video_info["transcript"]
    )
    topics = extract_result[0].payload["topics"]
    print(f"Found {len(topics)} topics")
    
    # Step 3: Process each topic concurrently
    print("\n--- Processing topics ---")
    
    # Create tasks for all topics with questions
    tasks = []
    topic_indices = []
    for i, topic in enumerate(topics):
        questions = [q for q in topic.get("questions", [])]
        if not questions:
            continue
        
        print(f"Queuing topic {i+1}/{len(topics)}: {topic['title']}")
        tasks.append(ctx.process_topic(
            topic_title=topic["title"],
            questions=questions,
            transcript=video_info["transcript"]
        ))
        topic_indices.append(i)
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Build processed topics list
    processed_topics = []
    for i, result in zip(topic_indices, results):
        processed = result[0].payload
        processed_topics.append({
            "original_title": topics[i]["title"],
            "rephrased_title": processed["rephrased_title"],
            "questions": processed["questions"]
        })
    
    # Step 4: Format output
    print("\n--- Generating Summary ---")
    output = f"""
# {video_info['title']}

**Video ID**: {video_info['video_id']}
**Thumbnail**: {video_info['thumbnail_url']}

---

"""
    for topic in processed_topics:
        output += f"## {topic['rephrased_title']}\n\n"
        for q in topic['questions']:
            output += f"### {q.get('rephrased', q.get('original', ''))}\n\n"
            output += f"{q.get('answer', '')}\n\n"
    
    print("\n" + "=" * 50)
    print("Processing completed successfully!")
    print("=" * 50 + "\n")
    
    return [Message(
        actor="youtube_summarizer",
        type="text",
        payload={
            "content": output,
            "video_info": video_info,
            "topics": processed_topics
        }
    )]


# --- App Context ---
class AppContext(BaseContext):
    def __init__(self, runner):
        super().__init__(runner)
        self.llm = self._bind(openai_llm_policy_v1)
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
    print("\n" + result[-1].payload['content'])
    
    # Optionally save to file
    with open("youtube_summary.md", "w") as f:
        f.write(result[-1].payload['content'])
    print("\nSummary saved to youtube_summary.md")
    
    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
