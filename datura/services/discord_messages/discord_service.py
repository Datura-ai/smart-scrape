import re
from typing import List
from datetime import datetime
from datura.services.discord_messages.discord_users_service import UserService
from datura.services.discord_messages.discord_pinecone_indexer import PineconeIndexer
from datura.services.discord_messages.database import MessageModel, Session, node_to_dict


class DiscordService:
    def __init__(self):
        self.userService = UserService()
        self.pinecone_indexer = PineconeIndexer()

    async def join_to_mentions(
        self, messages: List[MessageModel]
    ) -> List[MessageModel]:
        """
        Converts user mentions to readable user mentions by replacing <@user-id> with @username.
        Also generates message links if not already included in the message.
        """

        joined = []
        for message in messages:
            if message.content:
                mentions = re.findall(r"<@(\d+)>", message.content)
                for mention in mentions:
                    user = await self.userService.get_user(id=mention)
                    if user and (user.name or user.global_name):
                        replace = f"@{user.name or user.global_name or user.id}"
                        message.content = message.content.replace(
                            f"<@{mention}>", replace
                        )

            joined.append(message)

        return joined

    def extract_channels_from_query(self, query: str):
        channels = []

        # Pattern 1: #\[channel\](channel)
        pattern1 = r"#\[(\w+(-\w+)*)\]\(\w+(-\w+)*\)"
        matches1 = re.findall(pattern1, query)
        channels.extend([match[0] for match in matches1])

        # Pattern 2: #channel
        pattern2 = r"#(\w+(?:-\w+)*)"
        matches2 = re.findall(pattern2, query)
        channels.extend(matches2)

        # Remove channel names from the query text
        query = re.sub(pattern1, "", query)
        query = re.sub(pattern2, "", query)

        # Remove extra whitespace from the query text
        query = re.sub(r"\s+", " ", query).strip()

        return channels, query

    async def search(
        self,
        query,
        limit=10,
        possible_reply_limit=8,
        start_date=None,
        end_date=None,
    ) -> List[dict]:
        try:
            messages = []
            index_names, query = self.extract_channels_from_query(query)

            if index_names:
                nodes = await self.pinecone_indexer.retrieve_with_index_names(
                    query, limit, index_names, start_date, end_date,
                )
            else:
                nodes = await self.pinecone_indexer.retrieve(
                    query,
                    limit,
                    start_date,
                    end_date,
                )

            with Session() as session:
                for node in nodes:
                    message = node_to_dict(node)
                    date_str = message.get("date")

                    try:
                        date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                    except ValueError:
                        # If the first format fails, try parsing with the second format
                        date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S%z")

                    # Skip getting possible replies for messages in the rules, announcements and FAQ.
                    DISCORD_WELCOME_CHANNEL_IDS = [
                        "830068283314929684",
                        "830075335084474390",
                        "1215386737661055056",
                    ]

                    if message["channel_id"] not in DISCORD_WELCOME_CHANNEL_IDS:
                        possible_replies = (
                            session.query(MessageModel)
                            .filter(
                                MessageModel.channel_name == message["channel"],
                                MessageModel.created_at > date,
                            )
                            .order_by(MessageModel.created_at)
                            .limit(possible_reply_limit)
                            .all()
                        )

                        possible_replies = await self.join_to_mentions(possible_replies)

                        message["possible_replies"] = [
                            reply.to_map(only_parsable=True)
                            for reply in possible_replies
                        ]
                    else:
                        message["possible_replies"] = []

                    message["replies"] = [
                        {
                            "id": reply.get("id"),
                            "content": reply.get("content"),
                            "channel": reply.get("channel_name"),
                            "author": reply.get("author", reply.get("username")),
                            "date": reply.get("date"),
                            "author_url": reply.get("avatar"),
                            "link": f"https://discord.com/channels/799672011265015819/{reply.get('channel_id')}/{reply.get('id')}",
                        }
                        for reply in message["replies"]
                    ]

                    messages.append(message)

            return messages
        except Exception as e:
            raise Exception(f"Failed to execute search: {e}")
    pass
