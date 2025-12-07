import asyncio
from pocket_joe import (
    Message, 
    policy,
    BaseContext, 
    InMemoryRunner,
    OptionSchema,
    )
from examples.utils import openai_llm_policy_v1, web_seatch_ddgs_policy

# --- Tools ---
@policy.tool(description="Orchestrates LLM with web search tool")
async def search_agent(
    prompt: str,
    max_iterations: int = 3,
) -> list[Message]:
    """
    Orchestrator that gives the LLM access to web search.

    Args:
        prompt: The user prompt to process
        max_iterations: Maximum number of iterations to run

    Returns:
        List of Messages containing the conversation history with search results and final answer
    """

    system_message = Message(
        actor="system",
        type="text",
        payload={"content": "You are an AI assistant that can use tools to help answer user questions."}
    )
    prompt_message = Message(
        actor="user",
        type="text",
        payload={"content": prompt}
    )

    ctx = AppContext.get_ctx()
    history = [system_message, prompt_message]

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Search Agent Iteration {iteration} ---")
        selected_actions = await ctx.llm(
            observations=history, 
            options=OptionSchema.from_func([ctx.web_search])
            )
        history.extend(selected_actions)
        # stop if no tools called
        if not any(msg.type == "action_call" for msg in selected_actions):
            break

    return history

# --- App Context ---
class AppContext(BaseContext):

    def __init__(self, runner):
        super().__init__(runner)
        self.llm = self._bind(openai_llm_policy_v1)
        self.web_search = self._bind(web_seatch_ddgs_policy)
        self.search_agent = self._bind(search_agent)

# --- Main Execution ---

async def main():
    print("--- Starting Search Agent Demo ---")
    
    runner = InMemoryRunner()
    ctx = AppContext(runner)
    result = await ctx.search_agent(prompt="What is the latest Python version?")
    
    print(f"\nFinal Result: {result[-1].payload['content']}")
    print("--- Demo Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
