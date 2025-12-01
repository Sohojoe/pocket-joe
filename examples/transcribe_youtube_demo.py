"""Demo of YouTube transcription policy."""

import asyncio
from pocket_joe import BaseContext, InMemoryRunner
from examples.utils import TranscribeYouTubePolicy


class AppContext(BaseContext):
    def __init__(self, runner):
        super().__init__(runner)
        self.transcribe_youtube = self._bind(TranscribeYouTubePolicy)


async def main():
    # Example YouTube URL
    # url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    url = "https://youtu.be/h_Zk4fDDcSY?si=LaxkHlRgWTCzq1n5"
    
    print(f"Transcribing: {url}\n")
    
    runner = InMemoryRunner()
    ctx = AppContext(runner)
    
    result = await ctx.transcribe_youtube(url=url)
    payload = result[0].payload
    
    if "error" in payload:
        print(f"Error: {payload['error']}")
        return
    
    print(f"Video: {payload['title']}")
    print(f"Video ID: {payload['video_id']}")
    print(f"Thumbnail: {payload['thumbnail_url']}")
    print(f"Transcript length: {len(payload['transcript'])} chars")
    print(f"\nFirst 500 chars of transcript:")
    print(payload['transcript'][:500] + "...")


if __name__ == "__main__":
    asyncio.run(main())
