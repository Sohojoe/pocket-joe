import asyncio
from pocket_joe import (
    Message,
    policy,
    BaseContext,
    InMemoryRunner,
    OptionSchema,
    MessageBuilder,
    OptionCallPayload,
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

    system_builder = MessageBuilder(policy="system", role_hint_for_llm="system")
    system_builder.add_text("You are an AI assistant that can use tools to help answer user questions.")
    system_message = system_builder.to_message()

    prompt_builder = MessageBuilder(policy="user", role_hint_for_llm="user")
    prompt_builder.add_text(prompt)
    prompt_message = prompt_builder.to_message()

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
        # stop if no option calls
        if not any(msg.payload and isinstance(msg.payload, OptionCallPayload) for msg in selected_actions):
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

    # Get final text message
    final_msg = next((msg for msg in reversed(result) if msg.parts), '')
    print(f"\nFinal Result: {final_msg}")
    print("--- Demo Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
