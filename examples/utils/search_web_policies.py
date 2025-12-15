from ddgs import DDGS

from pocket_joe import policy

@policy.tool(description="Performs a web search and returns results.")
async def web_seatch_ddgs_policy(
    query: str,
) -> str:
    """
    Performs a web search and returns results.

    Args:
        query: The search query string to search for

    Returns:
        String containing formatted search results
    """

    results = DDGS().text(query, max_results=5) # type: ignore
    results_str = "\n\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])

    return results_str