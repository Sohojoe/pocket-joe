from contextvars import Context
from ddgs import DDGS

from pocket_joe import Action, Step
from pocket_joe import policy_spec_mcp_tool

@policy_spec_mcp_tool(
    description="Calls OpenAI GPT-4 with tool support",
)

@policy_spec_mcp_tool(description="Performs web search")
async def search_web_duckduckgo_policy(action: Action, ctx: Context, query: str) -> list[Step]:
    """
    Performs a web search and returns results.
    
    :param query: The search query string to search for
    """

    results = DDGS().text(query, max_results=5) # type: ignore
    # Convert results to a string
    results_str = "\n\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])
    
    return [
        Step(
            id="",  # Engine sets this
            actor=action.policy,
            type="action_result",
            payload={"content": results_str}
        )
    ]