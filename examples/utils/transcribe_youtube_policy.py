"""YouTube video transcription policy."""

import re
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

from pocket_joe import Message, Policy, policy_spec_mcp_tool


def _extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from URL."""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


@policy_spec_mcp_tool(description="Transcribe YouTube video to text")
class TranscribeYouTubePolicy(Policy):
    async def __call__(
        self,
        url: str,
    ) -> list[Message]:
        """
        Get video title, transcript and thumbnail from YouTube URL.
        
        :param url: YouTube video URL
        """
        video_id = _extract_video_id(url)
        if not video_id:
            return [
                Message(
                    id="",
                    actor=self.__class__.__name__,
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
                    actor=self.__class__.__name__,
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
                    actor=self.__class__.__name__,
                    type="action_result",
                    payload={"error": str(e)}
                )
            ]
