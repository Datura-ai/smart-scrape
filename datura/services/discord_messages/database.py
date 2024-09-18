import json
import os
from typing import Optional
from datetime import datetime
from discord import Message
from sqlalchemy import DateTime, ForeignKey
from sqlalchemy.orm import (
    mapped_column,
    Mapped,
    DeclarativeBase,
    relationship,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# NOTE ignore as discord is not used yet in production
# DB_URL = os.environ.get("DB_URL")
# if not DB_URL:
#     raise Exception("DB_URL is not set")

# engine = create_engine(DB_URL, echo=False)
# Session = sessionmaker(engine)

Session = None


class Base(DeclarativeBase):
    pass


class ChannelModel(Base):
    __tablename__ = "channels"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[Optional[str]]
    guild_id: Mapped[Optional[str]]
    guild_name: Mapped[Optional[str]]
    messages = relationship("MessageModel", back_populates="channel")

    def to_map(self):
        return {
            "id": self.id,
            "name": self.name,
            "guild_id": self.guild_id,
            "guild_name": self.guild_name,
        }


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[Optional[str]]
    global_name: Mapped[Optional[str]]
    avatar: Mapped[Optional[str]]
    messages = relationship("MessageModel", back_populates="user")

    def to_map(self):
        return {
            "id": self.id,
            "name": self.name,
            "global_name": self.global_name,
            "avatar": self.avatar,
        }


class MessageModel(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))
    user_avatar: Mapped[Optional[str]]
    user_name: Mapped[Optional[str]]
    content: Mapped[Optional[str]]
    guild_id: Mapped[Optional[str]]
    channel_id: Mapped[Optional[str]] = mapped_column(ForeignKey("channels.id"))
    channel_name: Mapped[Optional[str]]
    channel_nsfw: Mapped[Optional[bool]]
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    link: Mapped[Optional[str]]
    reference_id: Mapped[Optional[str]] = mapped_column(ForeignKey("messages.id"))
    user = relationship("UserModel", back_populates="messages")
    channel = relationship("ChannelModel", back_populates="messages")
    parent = relationship("MessageModel", foreign_keys=[reference_id], uselist=False)

    def to_map(self, only_parsable=False):
        if only_parsable:
            return {
                "id": self.id,
                "content": self.content,
                "channel": self.channel_name,
                "author": (
                    (self.user.global_name or self.user.name)
                    if self.user
                    else self.user_name
                ),
                "author_url": self.user_avatar,
                "link": self.link,
                "date": (
                    self.created_at.isoformat()
                    if self.created_at and isinstance(self.created_at, datetime)
                    else None
                ),
            }
        else:
            return {
                "id": self.id,
                "user_id": self.user_id,
                "user_name": self.user_name,
                "user_avatar": self.user_avatar,
                "content": self.content,
                "guild_id": self.guild_id,
                "channel_id": self.channel_id,
                "channel_nsfw": self.channel_nsfw,
                "created_at": (
                    self.created_at.isoformat()
                    if self.created_at and isinstance(self.created_at, datetime)
                    else None
                ),
                "link": self.link,
                "reference_id": self.reference_id,
            }


def message_to_dict(message: Message) -> dict:
    """Converts a discord.Message object into a dictionary capturing key details."""
    return {
        "id": str(message.id),
        "user_id": str(message.author.id),
        "user_avatar": (
            str(message.author.avatar.url) if message.author.avatar else None
        ),
        "content": str(message.content),
        "guild_id": str(message.guild.id) if message.guild else None,
        "channel_id": str(message.channel.id),
        "channel_nsfw": bool(message.channel.is_nsfw()),
        "created_at": str(message.created_at),
        "link": str(message.jump_url),
        "reference_id": (
            str(message.reference.message_id) if message.reference else None
        ),
    }


def message_to_channel_dict(message: Message) -> dict:
    """Converts a discord.Message object into a dictionary of a channel record"""
    return {
        "id": str(message.channel.id),
        "name": str(message),
        "guild_id": str(message.guild.id) if message.guild else None,
        "guild_name": str(message.guild.name) if message.guild else None,
    }


def message_to_user_dict(message: Message) -> dict:
    """Converts a discord.Message object into a dictionary of a user record"""
    return {
        "id": str(message.author.id),
        "name": str(message.author.name),
        "global_name": str(message.author.global_name),
        "avatar": str(message.author.avatar.url) if message.author.avatar else None,
    }


def node_to_dict(node) -> dict:
    channel_name = node.metadata.get("channel_name")
    channel_id = node.metadata.get("channel_id")
    author = node.metadata.get("author", node.metadata.get("username"))
    author_url = node.metadata.get("avatar")
    date = node.metadata.get("date")
    replies = json.loads(node.metadata.get("replies", "[]"))
    link = (
        f"https://discord.com/channels/799672011265015819/{channel_id}/{node.node.id_}"
    )

    return {
        "id": node.id_,
        "content": node.text,
        "channel": channel_name,
        "channel_id": channel_id,
        "author": author,
        "author_url": author_url,
        "replies": replies,
        "link": link,
        "date": date,
    }
