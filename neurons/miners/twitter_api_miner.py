from datura.protocol import TwitterAPISynapse, TwitterAPISynapseCall
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor


class TwitterAPIMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.client = TwitterScraperActor()

    async def execute_twitter_search(self, synapse: TwitterAPISynapse):
        if synapse.request_type == TwitterAPISynapseCall.GET_USER.value:
            response = await self.client.get_user_by_id(
                id=synapse.user_id,
            )
            synapse.results = response
        elif synapse.request_type == TwitterAPISynapseCall.GET_USER_WITH_USERNAME.value:
            response = await self.client.get_user_by_username(
                username=synapse.username,
            )
            synapse.results = response
        elif synapse.request_type == TwitterAPISynapseCall.GET_USER_FOLLOWINGS.value:
            response = await self.client.get_user_followings(
                id=synapse.user_id,
                maxUsersPerQuery=int(synapse.max_items) if synapse.max_items else 100,
            )
            synapse.results = response
        elif synapse.request_type == TwitterAPISynapseCall.SEARCH_TWEETS.value:
            response = await self.client.get_tweets_advanced(
                urls=None,
                searchTerms=[synapse.search_terms],
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
