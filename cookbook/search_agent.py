import asyncio
from pocket_joe import Action, Registry, loop_wrapper, invoke_action, InMemoryRunner

# Initialize Registry
registry = Registry()

# --- Tools ---

@registry.register("search_tool")
async def search_tool(action: Action, context) -> str:
    """
    Mock search tool.
    Input: Action(payload=query_string)
    Output: String result
    """
    query = action.payload
    print(f"  [Tool: search_tool] Searching for: '{query}'")
    await asyncio.sleep(0.5) # Simulate network latency
    return f"Search Results for '{query}': Python 3.13 released with GIL removal features..."

# --- Agent Policy ---

@registry.register("search_agent")
@loop_wrapper(max_turns=5)
@invoke_action()
async def search_agent(action: Action, context) -> dict:
    """
    A simple ReAct-style agent that searches if it doesn't know the answer.
    """
    query = action.payload
    history = action.history
    
    print(f"  [Agent] Thinking... (History Length: {len(history)})")

    # 1. Check history to see if we have results
    for item in history:
        if item["role"] == "tool" and item["name"] == "search_tool":
            # We have a result, so we can answer
            print("  [Agent] I have the info. Answering.")
            return {"done": True, "value": f"Based on search: {item['content']}"}

    # 2. If no results, decide to search
    print("  [Agent] I need to search.")
    return {
        "tool_call": "search_tool",
        "tool_args": query
    }

# --- Main Execution ---

async def main():
    print("--- Starting Search Agent Demo ---")
    
    runner = InMemoryRunner(registry)
    
    # Initial Action: User asks a question
    # We must define allowed edges (tools) for security/validity
    initial_action = Action(
        payload="What is the latest Python version?",
        edges=("search_tool",) 
    )
    
    result = await runner.execute("search_agent", initial_action)
    print(f"\nFinal Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
