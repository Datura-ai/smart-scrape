# How it works

The main class for the service is `discord_service`, which accepts query parameters such as query, start and end dates, limit of messages, and limit of each message's replies. This information is then passed to the `pinecone_indexer` class, which uses the `pinecone_indexes.json` file to configure retrievers. After retrieving data from Pinecone, the service queries the database to get the full body of messages from replies, replacing user IDs with nicknames using the `discord_users_service`.

- [database.py](../../datura/services/discord_messages/database.py)
- [discord_pinecone_indexer.py](../../datura/services/discord_messages/discord_pinecone_indexer.py)
- [pinecone_indexes.json](../../datura/services/discord_messages/pinecone_indexes.json)
- [discord_service.py](../../datura/services/discord_messages/discord_service.py)
- [discord_users_service.py](../../datura/services/discord_messages/discord_users_service.py)

### database.py

This file defines the `DISCORD_MESSAGES_DB_URL`, all models, and methods necessary for connecting to and retrieving data from the database. Messages are stored by the scraper bot "Datura Knight" from the Bittensor Discord channel and saved in a PostgreSQL database. The connection is established using the `DISCORD_MESSAGES_DB_URL` environment variable.

### discord_pinecone_indexer.py and pinecone_indexes.json

This file configures the retrieval strategy, defining how the service reads channels, gets retriever tools, and configures all retriever tools inside the router retriever. The `pinecone_indexes.json` file is crucial as it maps all vector stores to their namespaces and provides explanations for the LLM to retrieve based on the prompt.

### discord_service.py and discord_users_service.py

These files represent the top-level components of the service. The `discord_service` is used in the Discord search tool to initiate searches, calling the Pinecone indexer, retrieving data, and then fetching replies for each message. The `discord_users_service` handles replacing user IDs with nicknames.
