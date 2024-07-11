from datura.protocol import TwitterTweetSynapse
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor


class TwitterTweetMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.client = TwitterScraperActor()

    async def get_tweets(self, synapse: TwitterTweetSynapse):
        response = await self.client.get_tweets_advanced(
            urls=None,
            searchTerms=synapse.prompt.split(" "),
            start=synapse.start_date,
            end=synapse.end_date,
            maxItems=int(synapse.max_items) if synapse.max_items else 100,
            minimumRetweets=int(synapse.min_retweets) if synapse.min_retweets else None,
            minimumFavorites=int(synapse.min_likes) if synapse.min_likes else None,
            onlyVerifiedUsers=synapse.only_verified,
            onlyTwitterBlue=synapse.only_twitter_blue,
            onlyVideo=synapse.only_video,
            onlyImage=synapse.only_image,
            onlyQuote=synapse.only_quote,
        )
        synapse.results = response

        return synapse
