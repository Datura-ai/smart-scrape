from typing import List
from datura.protocol import TwitterScraperMedia, TwitterScraperTweet, TwitterScraperUser


def generalize_tweet_structure(tweets) -> List[dict]:
    """
    Converts the tweets data from its original structure to TwitterScraperTweet and returns a JSON version
    """
    modified_tweets = []
    users = tweets.get("includes", {}).get("users", [])
    medias = tweets.get("includes", {}).get("media", [])
    for tweet in tweets.get("data", []):
        user = next(
            (user for user in users if user.get("id") == tweet.get("author_id")), {}
        )
        parsed_medias = [
            TwitterScraperMedia(
                media_url=media.get("url", ""),
                type=media.get("type", ""),
            )
            for media in medias
        ]
        tweet = TwitterScraperTweet(
            id=tweet.get("id"),
            retweet_count=tweet.get("public_metrics").get("retweet_count"),
            reply_count=tweet.get("public_metrics").get("reply_count"),
            like_count=tweet.get("public_metrics").get("like_count"),
            quote_count=tweet.get("public_metrics").get("quote_count"),
            impression_count=tweet.get("public_metrics").get("impression_count"),
            bookmark_count=tweet.get("public_metrics").get("bookmark_count"),
            created_at=tweet.get("created_at"),
            text=tweet.get("text"),
            url=f"https://x.com/{user.get('username')}/status/{tweet.get('id')}",
            user=TwitterScraperUser(
                id=user.get("id"),
                username=user.get("username"),
                name=user.get("name"),
                created_at=user.get("created_at"),
                url=f"https://x.com/{user.get('username')}",
            ),
            media=parsed_medias,
        )
        modified_tweets.append(tweet.dict())
    return modified_tweets
