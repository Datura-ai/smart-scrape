from datura.protocol import TwitterAPISynapse, TwitterAPISynapseCall
from datura.services.twitter_api_wrapper import TwitterAPIClient


class TwitterAPIMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.client = TwitterAPIClient()

    async def get_user(self, synapse: TwitterAPISynapse):
        response = {}
        if synapse.request_type == TwitterAPISynapseCall.GET_USER:
            response_json, _, _ = await self.client.get_user(
                user_id=synapse.user_id,
                params={"user.fields": synapse.user_fields},
            )
            response = response_json
        elif synapse.request_type == TwitterAPISynapseCall.GET_USER_WITH_USERNAME:
            response_json, _, _ = await self.client.get_user_by_username(
                username=synapse.username,
                params={"user.fields": synapse.user_fields},
            )
            response = response_json
        elif synapse.request_type == TwitterAPISynapseCall.GET_USER_FOLLOWINGS:
            response_json, _, _ = await self.client.get_user_followings(
                user_id=synapse.user_id,
            )
            response = response_json

        synapse.results = response

        return synapse
