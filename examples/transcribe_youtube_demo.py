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

    info = await ctx.transcribe_youtube(url=url)

    if "error" in info:
        print(f"Error: {info['error']}")
        return

    print(f"Video: {info.get('title', 'Unknown')}")
    print(f"Video ID: {info.get('video_id', 'Unknown')}")
    print(f"Thumbnail: {info.get('thumbnail_url', 'Unknown')}")
    print(f"Transcript length: {len(info.get('transcript', ''))} chars")
    print(f"\nFull transcript:")
    print(info.get('transcript', ''))


if __name__ == "__main__":
    asyncio.run(main())
