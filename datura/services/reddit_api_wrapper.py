import asyncpraw
import os

REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")

if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
    raise ValueError("Reddit client ID and client secret are required")


class RedditAPIWrapper:
    async def search(self, query: str):
        async with asyncpraw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent="User-Agent: python: Gvardieli",
        ) as reddit:
            submission = await reddit.subreddit("all")

            posts = []

            async for post in submission.search(query):
                posts.append(
                    {
                        "subreddit": post.subreddit.display_name,
                        "title": post.title,
                        "url": post.url,
                    }
                )

            return posts
