# Discord Indexes Documentation

This document serves as the main reference for all documentation related to the Discord indexing service. It provides a comprehensive table of contents and explains key nuances. Note that this service is implemented on the miner side, so proper configuration of your Discord message client is essential for the miner to function correctly.

## Table of Contents

1. [Environment Variables](#env-variables)
2. [Required Packages](#required-packages)
3. [How it works](./how_it_works.md)
7. [Indexing Messages](./indexing_discord_messages.md)

## Env Variables

Setting environment variables is crucial for the implementation of Discord message indexing. The primary variable is `DISCORD_MESSAGES_DB_URL`, which points to the database where messages, users, and channels are stored.

### Setting Environment Variables

Follow the guidelines provided here in order to get and set environment variables: [env_variables.md](https://github.com/Datura-ai/smart-scrape/blob/main/docs/env_variables.md)

**DISCORD_MESSAGES_DB_URL**:
```
read_only_db_url
```

**PINECONE_API_KEY**:
```
750*****-****-****-****-************
```

**OPENAI_API_KEY**:
```
sk-************************************************
```

## Required Packages

The following packages are required to run the Discord message service. These packages are included in the project's `requirements.txt` file and will be installed automatically:

```
llama-index==0.10.28
llama-index-vector-stores-pinecone==0.1.4
llama-index-embeddings-openai>=0.1.5,<0.2.0
pinecone-client>=3.0.2,<4.0.0
discord.py
psycopg2-binary
SQLAlchemy
openai
python-dateutil
pytz
```
