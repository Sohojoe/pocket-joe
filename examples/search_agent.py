import asyncio
# from pocket_joe import Action, Step, Registry, Context, InMemoryRunner, loop_wrapper, invoke_action, policy_spec
from pocket_joe import (
    Action, Step, policy_spec,
    Registry, Context, 
    InMemoryRunner, 
    # loop_wrapper, invoke_action,
    )
from dataclasses import replace
from examples.utils.llm_policies import openai_llm_policy_v1

# --- Tools ---

@policy_spec(description="Performs web search")
async def web_search_policy(action: Action, ctx: Context) -> list[Step]:
    """
    Performs a web search and returns results.
    Input: action.payload = {"query": "..."}
    Output: action_result with search results
    """
    query = action.payload["query"]
    
    # Call search API (e.g., Bing, Google, Brave)
    results = await search_api.query(query)  # Mock
    
    return [
        Step(
            id="",  # Engine sets this
            actor="web_search_policy",
            type="action_result",
            payload={"text": results}
        )
    ]

@policy_spec(description="Orchestrator with LLM and search")
async def search_agent(action: Action, ctx: Context) -> list[Step]:
    """
    Orchestrator that gives the LLM access to web search.
    """
    # Build sub-action for LLM
    # payload['ledger'] = ctx.get_ledger()  # Pass conversation history to LLM
    messages = [{"role": "system", "content": "You are an AI assistant that can use tools to help answer user questions."}]
    messages.append({"role": "user", "content": action.payload.get("text", "")})
    if 'messages' in action.payload:
        messages.extend(action.payload['messages'])
    payload = {**action.payload}
    payload['messages'] = messages

    llm_action = Action(
        policy="llm_policy",
        payload=payload,
        actions=action.actions | {"web_search_policy"},
    )
    
    # Call LLM with decorators: loop until done, auto-execute tool calls
    steps = await ctx.call(
        action=llm_action,
        # decorators=[loop_wrapper(max_turns=5), invoke_action()]
    )
    
    return steps


# --- Main Execution ---

async def main():
    print("--- Starting Search Agent Demo ---")
    
    # Initialize Registry with all policies
    # Option 1: Pass all policies to constructor
    registry = Registry(web_search_policy, search_agent)
    # Option 2: Override imported policy name
    registry.register_policy(openai_llm_policy_v1, alias="llm_policy")
    
    runner = InMemoryRunner(registry)
    
    # Initial Action: User asks a question
    # We must define allowed edges (tools) for security/validity
    initial_action = Action(
        policy="search_agent",
        payload= {
            "text": "What is the latest Python version?"
        },
    )
    
    result = await runner.execute(initial_action)
    print(f"\nFinal Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
