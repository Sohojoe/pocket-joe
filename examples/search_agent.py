import asyncio
import os
from pocket_joe import (
    Message,
    policy,
    BaseContext,
    InMemoryRunner,
    OptionSchema,
    MessageBuilder,
    OptionCallPayload,
)
from examples.utils import CompletionsAdapter, web_seatch_ddgs_policy


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
    client = CompletionsAdapter.client()
    response = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=adapter.messages,  # type: ignore[arg-type]
        tools=adapter.tools or [],  # type: ignore[arg-type]
    )
    return adapter.decode(response, policy="llm_policy")


# --- Tools ---
@policy.tool(description="Orchestrates LLM with web search tool")
async def search_agent(
    prompt: str,
    max_iterations: int = 3,
) -> list[Message]:
    """Orchestrator that gives the LLM access to web search.

    Args:
        prompt: The user prompt to process.
        max_iterations: Maximum number of iterations to run.

    Returns:
        List of Messages containing conversation history with search results.
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
        self.llm = self._bind(llm_policy)
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
