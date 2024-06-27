import bittensor as bt
from datura.protocol import TwitterAPISynapse, TwitterAPISynapseCall
from datura.services.twitter_api_wrapper import TwitterAPIClient


class TwitterAPIMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.client = TwitterAPIClient()

    def define_params(self, synapse: TwitterAPISynapse):
        params = {}
        if synapse.user_fields:
            params["user.fields"] = synapse.user_fields

        if synapse.expansions:
            params["expansions"] = synapse.expansions

        if synapse.max_results:
            params["max_results"] = synapse.max_results

        if synapse.pagination_token:
            params["pagination_token"] = synapse.pagination_token

        if synapse.tweet_fields:
            params["tweet.fields"] = synapse.tweet_fields

        return params

    async def get_user(self, synapse: TwitterAPISynapse):
        if synapse.request_type == TwitterAPISynapseCall.GET_USER.value:
            params = self.define_params(synapse)
            response, _, _ = await self.client.get_user(
                user_id=synapse.user_id,
                params=params,
            )
            synapse.results = response
        elif synapse.request_type == TwitterAPISynapseCall.GET_USER_WITH_USERNAME.value:
            params = self.define_params(synapse)
            response, _, _ = await self.client.get_user_by_username(
                username=synapse.username,
                params=params,
            )
            synapse.results = response
        elif synapse.request_type == TwitterAPISynapseCall.GET_USER_FOLLOWINGS.value:
            params = self.define_params(synapse)
            response, _, _ = await self.client.get_user_followings(
                user_id=synapse.user_id,
                params=params,
            )
            synapse.results = response

        return synapse
