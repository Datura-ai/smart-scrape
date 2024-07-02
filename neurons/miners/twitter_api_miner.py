from datura.protocol import TwitterAPISynapse, TwitterAPISynapseCall
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor


class TwitterAPIMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.client = TwitterScraperActor()

    async def get_user(self, synapse: TwitterAPISynapse):
        max_users_per_query = synapse.max_users_per_query
        if max_users_per_query:
            max_users_per_query = int(max_users_per_query)
        if synapse.request_type == TwitterAPISynapseCall.GET_USER.value:
            response = await self.client.get_user_by_id(
                id=synapse.user_id,
                maxUsersPerQuery=max_users_per_query,
            )
            synapse.results = response
        elif synapse.request_type == TwitterAPISynapseCall.GET_USER_WITH_USERNAME.value:
            response = await self.client.get_user_by_username(
                username=synapse.username,
                maxUsersPerQuery=max_users_per_query,
            )
            synapse.results = response
        elif synapse.request_type == TwitterAPISynapseCall.GET_USER_FOLLOWINGS.value:
            response = await self.client.get_user_followings(
                id=synapse.user_id,
                maxUsersPerQuery=max_users_per_query,
            )
            synapse.results = response

        return synapse
