import os
from typing import List

# DEALINGS IN THE SOFTWARE.p
import traceback
import bittensor as bt
from apify_client import ApifyClientAsync
from template.protocol import (
    TwitterScraperTweet,
    TwitterScraperMedia,
    TwitterScraperUser,
)


APIFY_API_KEY = os.environ.get("APIFY_API_KEY")

# todo at ths moment just warning, later it will be required
if not APIFY_API_KEY:
    raise ValueError(
        "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md"
    )


class TwitterScraperActor:
    def __init__(self) -> None:
        # Actor: https://apify.com/apidojo/tweet-scraper
        self.actor_id = "61RPP7dywgiy0JPD0"
        self.client = ApifyClientAsync(token=APIFY_API_KEY)

    async def get_tweets(
        self, urls: List[str], add_user_info: bool = True
    ) -> List[TwitterScraperTweet]:
        if not APIFY_API_KEY:
            bt.logging.warning(
                "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md. This will be required in the next release."
            )
            return []
        try:
            run_input = {
                "startUrls": urls,
            }

            run = await self.client.actor(self.actor_id).call(run_input=run_input)

            tweets: List[TwitterScraperTweet] = []

            async for item in self.client.dataset(
                run["defaultDatasetId"]
            ).iterate_items():
                media_list = item.get("extendedEntities", {}).get("media", [])

                media_list = [
                    TwitterScraperMedia(
                        media_url=media.get("media_url_https"), type=media.get("type")
                    )
                    for media in media_list
                ]

                author = item.get("author", {})

                tweet = TwitterScraperTweet(
                    user=TwitterScraperUser(
                        id=author.get("id"),
                        created_at=author.get("createdAt"),
                        description=author.get("description"),
                        followers_count=author.get("followers"),
                        favourites_count=author.get("favouritesCount"),
                        media_count=author.get("mediaCount"),
                        statuses_count=author.get("statusesCount"),
                        verified=author.get("isVerified"),
                        profile_image_url=author.get("profilePicture"),
                        url=author.get("url"),
                        name=author.get("name"),
                        username=author.get("userName"),
                    ),
                    id=item.get("id"),
                    full_text=item.get("text"),
                    reply_count=item.get("replyCount"),
                    retweet_count=item.get("retweetCount"),
                    like_count=item.get("likeCount"),
                    quote_count=item.get("quoteCount"),
                    url=item.get("url"),
                    created_at=item.get("createdAt"),
                    is_quote_tweet=item.get("isQuote"),
                    is_retweet=item.get("isRetweet"),
                    media=media_list,
                )

                tweets.append(tweet)

            return tweets
        except Exception as e:
            error_message = f"Twitter Scraper Actor: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.warning("\n".join(tb_str) + error_message)
            return []
