import os
import re
import sys
import json
from typing import List
from datetime import datetime
from sqlalchemy.orm import aliased
from sqlalchemy.orm import joinedload
from llama_index.core import Settings
from pinecone import Pinecone, PodSpec
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from datura.services.discord_messages.database import (
    MessageModel,
    ChannelModel,
    UserModel,
    Session,
)


# For each object, the structure consists of the index name and nested namespaces inside.
# The namespaces' values are the channel IDs, and you can also include the last message
# ID by combining it with the channel ID, like so: channel_id:message_id.
channels_grouper = {
    "discord-subnets": {
        "alpha-1": "1161764867166961704",
        "beta-2": "1161764868265869314",
        "gamma-3": "1222226314824777853",
        "delta-4": "1161765008347254915",
        "epsilon-5": "1214225551364202496",
        "zeta-6": "1200530530416988353",
        "eta-7": "1215311984799653918",
        "theta-8": "1162384774170677318",
        "iota-9": "1162768567821930597",
        "kappa-10": "1163969538191269918",
        "lambda-11": "1161765231953989712",
        "mu-12": "1201941624243109888",
        "nu-13": "1185617142914236518",
        "xi-14": "1182422353360195695",
        "omicron-15": "1166816300962693170",
        "pi-16": "1166816341697761300",
        "rho-17": "1173712344409460766",
        "sigma-18": "1172669887697653881",
        "tau-19": "1186691482749505627",
        "upsilon-20": "1194736998250975332",
        "phi-21": "1182096085636878406",
        "chi-22": "1189589759065067580",
        "psi-23": "1191833510021955695",
        "omega-24": "1214246819886931988",
        "alef-25": "1174839377659183174",
        "bet-26": "1178397855053000845",
        "gimel-27": "1174835090539433994",
        "dalet-28": "1211711222421000273",
        "he-29": "1207399745031770152",
        "wav-30": "1211034110722842674",
        "zayin-31": "1222990781577433232",
        "chet-32": "1215319932062011464",
    },
    "discord-welcome": {
        "rules": "830068283314929684",
        "announcements": "830075335084474390",
        "faq": "1215386737661055056",
    },
    "discord-community": {
        "welcome": ["830068283314929684", "830075335084474390", "1215386737661055056"],
        "general": "799672011814862902",
    }
}


class DiscordMessagesIndex:
    def __init__(self):
        self.index_path = f"{os.path.abspath(os.getcwd())}/llama_index"
        self.OPENAI_APIKEY = os.environ.get("OPENAI_API_KEY")
        self.PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        Settings.embed_model = OpenAIEmbedding(
            api_key=self.OPENAI_APIKEY, model="text-embedding-3-small", embed_batch_size=1000
        )
        self.users = self.get_users()
        self.channels = self.get_channels()

    def get_users(self):
        users = {}
        with Session() as session:
            user_models = session.query(UserModel).all()

        users = {user.id: user for user in user_models}
        return users

    def get_channels(self):
        channels = {}

        with Session() as session:
            channel_models = session.query(ChannelModel).all()

        channels = {channel.id: channel for channel in channel_models}
        return channels

    def parse_argument(self, arg: str):
        parts = arg.split(":")
        id = parts[0]
        after = parts[1] if len(parts) > 1 else None
        return id, after

    def read_messages(self, channel_id: str, after: str | None):
        with Session() as session:
            ParentMessage = aliased(MessageModel)

            query = (
                session.query(MessageModel)
                .outerjoin(MessageModel.user)
                .outerjoin(MessageModel.channel)
                .outerjoin(ParentMessage, MessageModel.reference_id == ParentMessage.id)
                .filter(MessageModel.channel_id == channel_id)
                .order_by(MessageModel.created_at)
                .options(
                    joinedload(MessageModel.user),
                    joinedload(MessageModel.channel),
                    joinedload(MessageModel.parent),
                )
            )

            all_messages = query.all()
            print(
                f">>> channel_id: {channel_id} all messages: {len(all_messages)}")

            messages = []
            if after:
                after_index = next(
                    (i for i, m in enumerate(all_messages) if m.id == after), None
                )
                if after_index is not None:
                    messages = all_messages[after_index + 1:]
                print(
                    f">>> channel_id: {channel_id} after:{after} messages: {len(messages)}"
                )
            else:
                messages = all_messages

            return messages

    def parse_channel_name(self, channel_name: str):
        if not channel_name:
            return channel_name

        channel_match = re.search(
            r"(<Thread id=\d+ name='([^']+)'|<TextChannel id=\d+ name='([^']+)')",
            channel_name,
        )

        if channel_match:
            if channel_match.group(2):
                channel_name = channel_match.group(2)
            else:
                channel_name = channel_match.group(3)

        return channel_name

    def replace_mentions_in_message_content(self, content: str, message_channel_name: str) -> str:
        user_mentions = re.findall(r"<@(\d+)>", content)

        # Handle user mentions in the format <@1189589759065067580>
        for mention in user_mentions:
            user = self.users.get(mention)
            if user and (user.global_name or user.name):
                replace = f"@{user.global_name or user.name or user.id}"
                content = content.replace(f"<@{mention}>", replace)

        # Handle channel mentions in the format <@&1189589759065067580>
        channel_mentions = re.findall(r"<@&(\d+)>", content)
        for mention in channel_mentions:
            content = content.replace(
                f"<@&{mention}>",
                f"@{message_channel_name}",
            )

        # Handle channel mentions in the format <#1189589759065067580>
        channel_mentions = re.findall(r"<#(\d+)>", content)
        for mention in channel_mentions:
            content = content.replace(
                f"<#{mention}>",
                f"#{message_channel_name}",
            )

        # Replace channel links with channel names
        channel_links = re.findall(
            r"(https://discord.com/channels/\d+/\d+(/\d+)?)", content
        )
        for channel_link in channel_links:
            channel_url = channel_link[0]
            channel_ids = channel_url.split("/")

            channel_ids = channel_url.replace("https://discord.com/channels/", "").split(
                "/"
            )

            if len(channel_ids) == 3:
                channel_id = channel_ids[1]
            else:
                channel_id = channel_ids[len(channel_ids) - 1]

            channel = self.channels.get(channel_id)

            if channel:
                channel_name = self.parse_channel_name(channel.name)
                content = content.replace(
                    channel_url,
                    f"[#{channel_name}]({channel_url})",
                )

        return content

    def parse_to_text_nodes(self, messages: List[MessageModel]):
        if len(messages) == 0:
            return []

        nodes = {}
        message_nodes = []

        for _, message in enumerate(messages):
            user = self.users.get(message.user_id)

            channel_name = self.parse_channel_name(
                message.channel.name if message.channel else message.channel_name
            )

            metadata = {
                "channel_name": channel_name,
                "channel_id": message.channel_id,
                "username": message.user.name if message.user else message.user_name,
                "avatar": message.user_avatar,
                "date": (
                    message.created_at.isoformat()
                    if message.created_at and isinstance(message.created_at, datetime)
                    else None
                ),
                "timestamp_date": (
                    int(message.created_at.timestamp())
                    if message.created_at and isinstance(message.created_at, datetime)
                    else None
                ),
            }

            if user and user.global_name and user.global_name != "None":
                metadata["author"] = user.global_name

            metadata = {k: v for k, v in metadata.items() if v is not None}

            content = self.replace_mentions_in_message_content(
                message.content, channel_name)
            node = TextNode(id_=message.id, text=content, metadata=metadata)
            nodes[node.id_] = node

            if message.reference_id:
                parent_node: TextNode = nodes.get(message.reference_id)
                if parent_node:
                    reply = {
                        "id": message.id,
                        "content": content,
                        "channel_name": channel_name,
                        "channel_id": message.channel_id,
                        "username": (
                            message.user.name if message.user else message.user_name
                        ),
                        "avatar": message.user_avatar,
                        "date": (
                            message.created_at.isoformat()
                            if message.created_at
                            and isinstance(message.created_at, datetime)
                            else None
                        ),
                    }

                    if user and user.global_name and user.global_name != "None":
                        reply["author"] = user.global_name

                    reply = {k: v for k, v in reply.items() if v is not None}
                    replies_json = parent_node.metadata.get("replies")

                    if not replies_json:
                        parent_node.metadata["replies"] = json.dumps([reply])
                    else:
                        replies = json.loads(replies_json)
                        replies.append(reply)
                        parent_node.metadata["replies"] = json.dumps(replies)

            message_nodes.append(node)

        return message_nodes

    def parse_channel_grouper(self):
        index_ntn = {}

        for index, values in channels_grouper.items():
            ntn = {}

            for namespace, channel_id in values.items():
                if isinstance(channel_id, list):
                    nodes = []
                    for cid in channel_id:
                        id, after = self.parse_argument(cid)
                        messages = self.read_messages(id, after)
                        for n in self.parse_to_text_nodes(messages):
                            nodes.append(n)
                    ntn[namespace] = nodes
                else:
                    id, after = self.parse_argument(channel_id)
                    messages = self.read_messages(id, after)
                    nodes = self.parse_to_text_nodes(messages)
                    ntn[namespace] = nodes

            index_ntn[index] = ntn

        return index_ntn

    def create_indexes(self, create=True):
        index_values = self.parse_channel_grouper()
        for index_name, values in index_values.items():
            if create:
                self.pc.create_index(
                    index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=PodSpec(
                        environment="us-east-1-aws",
                        pod_type="p1.x1",
                    ),
                )

            pinecone_index = self.pc.Index(index_name)
            for namespace, nodes in values.items():
                print(f">>> Deleting Namespace: {namespace}")
                pinecone_index.delete(namespace=namespace, delete_all=True)
                print(f">>> Namespace Deleted: {namespace}")
                vector_store = PineconeVectorStore(
                    pinecone_index=pinecone_index,
                    api_key=self.PINECONE_API_KEY,
                    namespace=namespace,
                )
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store)
                index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=storage_context,
                    show_progress=True,
                )

                index.set_index_id(index_name)
                index.storage_context.persist(
                    persist_dir=f"{self.index_path}/index")


if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv:
        print(
            "Usage: python3 datura/scripts/discord_messages_indexer.py [--ignore-creating-indexes]")
        sys.exit(1)

    create_indexes = not ("--ignore-creating-indexes" in sys.argv)
    indexer = DiscordMessagesIndex()
    indexer.create_indexes(create=create_indexes)
