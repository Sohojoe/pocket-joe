"""YouTube video transcription policy."""

import re
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

from pocket_joe import Message, policy


def _extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from URL."""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


@policy.tool(description="Transcribe YouTube video and retrieve transcript and metadata")
async def transcribe_youtube_policy(
    url: str,
) -> list[Message]:
    """
    Get video title, transcript and thumbnail from YouTube URL.
    
    Args:    
        url: YouTube video URL

    Returns:
        List containing a single Message with payload: {title, transcript, thumbnail_url, video_id}
        or {error} on failure
    """
    video_id = _extract_video_id(url)
    if not video_id:
        return [
            Message(
                id="",
                actor="transcribe_youtube_policy",
                type="action_result",
                payload={"error": "Invalid YouTube URL"}
            )
        ]

    try:
        # Get title using BeautifulSoup
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('title')
        title = title_tag.text.replace(" - YouTube", "") if title_tag else "Unknown Title"
        
        # Get thumbnail
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        
        # Get transcript
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        transcript = " ".join([snippet.text for snippet in fetched_transcript])
        
        return [
            Message(
                id="",
                actor="transcribe_youtube_policy",
                type="action_result",
                payload={
                    "title": title,
                    "transcript": transcript,
                    "thumbnail_url": thumbnail_url,
                    "video_id": video_id
                }
            )
        ]
    except Exception as e:
        return [
            Message(
                id="",
                actor="transcribe_youtube_policy",
                type="action_result",
                payload={"error": str(e)}
            )
        ]
