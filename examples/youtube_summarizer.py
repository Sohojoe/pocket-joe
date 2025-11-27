import asyncio
from pocket_joe import Action, Registry, Context, InMemoryRunner, loop_wrapper, invoke_action

registry = Registry()

# --- Tools ---

@registry.register("get_transcript")
async def get_transcript(action: Action, context) -> str:
    url = action.payload
    print(f"  [Tool: get_transcript] Fetching transcript for: {url}")
    await asyncio.sleep(0.5)
    return "This is a mock transcript of the video. It talks about AI agents and PocketJoe..."

@registry.register("summarize_text")
async def summarize_text(action: Action, context) -> str:
    text = action.payload
    print(f"  [Tool: summarize_text] Summarizing text length: {len(text)}")
    await asyncio.sleep(0.5)
    return "Summary: The video explains how PocketJoe simplifies agent development using Policies and Actions."

# --- Agent Policy ---

@registry.register("youtube_agent")
@loop_wrapper(max_turns=5)
@invoke_action()
async def youtube_agent(action: Action, context) -> dict:
    history = action.history
    url = action.payload # Initial payload is URL
    
    # 1. Check if we have a summary
    for item in history:
        if item["role"] == "tool" and item["name"] == "summarize_text":
            return {"done": True, "value": item["content"]}
            
    # 2. Check if we have a transcript
    transcript = None
    for item in history:
        if item["role"] == "tool" and item["name"] == "get_transcript":
            transcript = item["content"]
            break
            
    if transcript:
        print("  [Agent] I have the transcript. Now summarizing.")
        return {
            "tool_call": "summarize_text",
            "tool_args": transcript
        }
    else:
        print("  [Agent] I need the transcript.")
        return {
            "tool_call": "get_transcript",
            "tool_args": url
        }

# --- Main ---

async def main():
    print("--- Starting YouTube Summarizer Demo ---")
    runner = InMemoryRunner(registry)
    
    initial_action = Action(
        payload="https://youtube.com/watch?v=12345",
        edges=("get_transcript", "summarize_text")
    )
    
    result = await runner.execute("youtube_agent", initial_action)
    print(f"\nFinal Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
