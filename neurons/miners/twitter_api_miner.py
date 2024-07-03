from datura.protocol import TwitterAPISynapse, TwitterAPISynapseCall
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor


class TwitterAPIMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.client = TwitterScraperActor()

    async def execute_twitter_search(self, synapse: TwitterAPISynapse):
        max_items = synapse.max_items
        if max_items:
            max_items = int(max_items)
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
                maxUsersPerQuery=max_items,
            )
            synapse.results = response

        return synapse
