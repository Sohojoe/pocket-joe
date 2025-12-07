from ddgs import DDGS

from pocket_joe import Message, policy

@policy.tool(description="Performs a web search and returns results.")
async def web_seatch_ddgs_policy(
    query: str,
) -> list[Message]:
    """
    Performs a web search and returns results.
        
    Args:
        query: The search query string to search for

    Returns:
        List containing a single Message with payload containing formatted search results
    """

    results = DDGS().text(query, max_results=5) # type: ignore
    # Convert results to a string
    results_str = "\n\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])
    
    return [
        Message(
            id="",  # Engine sets this
            actor="web_seatch_ddgs_policy",
            type="action_result",
            payload={"content": results_str}
        )
    ]