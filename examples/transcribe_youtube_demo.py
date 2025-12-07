"""Demo of YouTube transcription policy."""

import asyncio
import sys
from pocket_joe import BaseContext, InMemoryRunner
from examples.utils import transcribe_youtube_policy


class AppContext(BaseContext):
    def __init__(self, runner):
        super().__init__(runner)
        self.transcribe_youtube = self._bind(transcribe_youtube_policy)


async def main():
    # Get URL from command line or use default
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
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
    print(f"\nFull transcript:")
    print(payload['transcript'])


if __name__ == "__main__":
    asyncio.run(main())
