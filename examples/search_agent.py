import asyncio
# from pocket_joe import Action, Message, Registry, Context, InMemoryRunner, loop_wrapper, invoke_action, policy_spec
from pocket_joe import (
    Message, 
    policy_spec_mcp_resource, policy_spec_mcp_tool,
    BaseContext, 
    InMemoryRunner, 
    Policy,
    # loop_wrapper, invoke_action,
    )
from dataclasses import replace
from examples.utils import OpenAILLMPolicy_v1, WebSeatchDdgsPolicy

# --- Tools ---
@policy_spec_mcp_tool(description="Orchestrator with LLM and search")
class SearchAgent(Policy):
    ctx: "AppContext" # ocerride to specify context type
    async def __call__(
        self,
        prompt: str,
        max_iterations: int = 3,
    ) -> list[Message]:
        """
        Orchestrator that gives the LLM access to web search.
        :param prompt: The user prompt to process
        :param max_iterations: Maximum number of iterations to run
        """
        # Build sub-action for LLM
        # payload['ledger'] = ctx.get_ledger()  # Pass conversation history to LLM
        # payload = [{"role": "system", "content": "You are an AI assistant that can use tools to help answer user questions."}]
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

        history = [system_message, prompt_message]

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Search Agent Iteration {iteration} ---")
            selected_actions = await self.ctx.llm(observations=history, options=["web_search"])
            history.extend(selected_actions)
            # stop of no tools called
            if not any(msg.type == "action_call" for msg in selected_actions):
                break
    
        return history

# --- App Context ---
class AppContext(BaseContext):

    def __init__(self, runner):
        super().__init__(runner)
        # self.llm = self._bind(OpenAILLMPolicy_v1)
        # self.web_search = self._bind(WebSeatchDdgsPolicy)
        # self.search_agent = self._bind(SearchAgent)
        self.llm = self._bind(OpenAILLMPolicy_v1)
        self.web_search = self._bind(WebSeatchDdgsPolicy)
        self.search_agent = self._bind(SearchAgent)

# --- Main Execution ---

async def main():
    print("--- Starting Search Agent Demo ---")
    
    # Initialize Registry with all policies
    # registry = Registry(search_agent)
    # registry.register_policy(openai_llm_policy_v1, alias="llm_policy")
    # registry.register_policy(search_web_duckduckgo_policy, alias="search_web_policy")
    
    runner = InMemoryRunner()
    ctx = AppContext(runner)
    result = await ctx.search_agent(prompt="What is the latest Python version?")
    
    print(f"\nFinal Result: {result[-1].payload['content']}")
    print("--- Demo Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
