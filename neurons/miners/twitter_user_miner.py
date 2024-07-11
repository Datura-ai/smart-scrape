from datura.protocol import TwitterUserSynapse, TwitterAPISynapseCall
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor


class TwitterUserMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.client = TwitterScraperActor()

    async def get_user(self, synapse: TwitterUserSynapse):
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
        elif synapse.request_type == TwitterAPISynapseCall.GET_USER_FOLLOWERS.value:
            response = await self.client.get_user_followers(
                id=synapse.user_id,
                maxUsersPerQuery=int(synapse.max_items) if synapse.max_items else 100,
            )
            synapse.results = response

        return synapse
