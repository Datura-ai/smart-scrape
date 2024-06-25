import bittensor as bt
from datura.protocol import TwitterAPISynapse, TwitterAPISynapseCall
from datura.services.twitter_api_wrapper import TwitterAPIClient


class TwitterAPIMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.client = TwitterAPIClient()

    async def get_user(self, synapse: TwitterAPISynapse):
        try:
            if synapse.request_type == TwitterAPISynapseCall.GET_USER.value:
                response, _, _ = await self.client.get_user(
                    user_id=synapse.user_id,
                    params={"user.fields": synapse.user_fields},
                )
                synapse.results = response
            elif synapse.request_type == TwitterAPISynapseCall.GET_USER_WITH_USERNAME.value:
                response, _, _ = await self.client.get_user_by_username(
                    username=synapse.username,
                    params={"user.fields": synapse.user_fields},
                )
            elif synapse.request_type == TwitterAPISynapseCall.GET_USER_FOLLOWINGS.value:
                response, _, _ = await self.client.get_user_followings(
                    user_id=synapse.user_id,
                )
                synapse.results = response
        except Exception as e:
            bt.logging.error(f"error in e get_user {e}")
            pass

        return synapse
