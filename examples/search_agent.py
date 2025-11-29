import asyncio
# from pocket_joe import Action, Step, Registry, Context, InMemoryRunner, loop_wrapper, invoke_action, policy_spec
from pocket_joe import (
    Action, Step, 
    policy_spec_mcp_resource, policy_spec_mcp_tool,
    Registry, Context, 
    InMemoryRunner, 
    # loop_wrapper, invoke_action,
    )
from dataclasses import replace
from examples.utils import openai_llm_policy_v1, search_web_duckduckgo_policy

# --- Tools ---

@policy_spec_mcp_resource(description="Orchestrator with LLM and search")
async def search_agent(action: Action, ctx: Context) -> list[Step]:
    """
    Orchestrator that gives the LLM access to web search.
    """
    # Build sub-action for LLM
    # payload['ledger'] = ctx.get_ledger()  # Pass conversation history to LLM
    # payload = [{"role": "system", "content": "You are an AI assistant that can use tools to help answer user questions."}]
    system_step = Step(
        actor="system",
        type="text",
        payload={"content": "You are an AI assistant that can use tools to help answer user questions."}
    )
    payload = [system_step] + action.payload

    llm_action = Action(
        policy="llm_policy",
        payload=payload,
        actions=action.actions | {"search_web_policy"},
    )
    
    # Call LLM with decorators: loop until done, auto-execute tool calls
    steps = await ctx.call(
        action=llm_action,
        # decorators=[loop_wrapper(max_turns=5), invoke_action()]
    )

    web_action = Action(
        policy="search_web_policy",
        payload=steps,
    )
    steps = await ctx.call(action=web_action)

    return steps


# --- Main Execution ---

async def main():
    print("--- Starting Search Agent Demo ---")
    
    # Initialize Registry with all policies
    registry = Registry(search_agent)
    registry.register_policy(openai_llm_policy_v1, alias="llm_policy")
    registry.register_policy(search_web_duckduckgo_policy, alias="search_web_policy")
    
    runner = InMemoryRunner(registry)
    
    # Initial Action: User asks a question
    # We must define allowed edges (tools) for security/validity
    first_step = Step(
        actor="user",
        type="text",
        payload={"content": "What is the latest Python version?"}
    )
    initial_action = Action(
        policy="search_agent",
        payload= [first_step],
    )
    
    result = await runner.execute(initial_action)
    print(f"\nFinal Result: {result[0].payload['content']}")

if __name__ == "__main__":
    asyncio.run(main())
