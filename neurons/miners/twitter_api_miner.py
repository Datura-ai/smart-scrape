import bittensor as bt
from datura.protocol import TwitterAPISynapse, TwitterAPISynapseCall
from datura.services.twitter_api_wrapper import TwitterAPIClient


class TwitterAPIMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.client = TwitterAPIClient()

    async def get_user(self, synapse: TwitterAPISynapse):
        if synapse.request_type == TwitterAPISynapseCall.GET_USER.value:
            params = {}
            if synapse.user_fields:
                params["user.fields"] = synapse.user_fields
            response, _, _ = await self.client.get_user(
                user_id=synapse.user_id,
                params=params,
            )
            synapse.results = response
        elif synapse.request_type == TwitterAPISynapseCall.GET_USER_WITH_USERNAME.value:
            params = {}
            if synapse.user_fields:
                params["user.fields"] = synapse.user_fields
            response, _, _ = await self.client.get_user_by_username(
                username=synapse.username,
                params=params,
            )
            synapse.results = response
        elif synapse.request_type == TwitterAPISynapseCall.GET_USER_FOLLOWINGS.value:
            params = {}
            if synapse.user_fields:
                params["user.fields"] = synapse.user_fields
            response, _, _ = await self.client.get_user_followings(
                user_id=synapse.user_id,
                params=params,
            )
            synapse.results = response

        return synapse
