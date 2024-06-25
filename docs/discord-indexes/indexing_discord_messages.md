# Indexing Discord Messages

Before diving into the script, let's understand how we manage the data. Messages are stored in a public database accessible to everyone. This is achieved by a scraper bot running live and listening to the conversations on the Bittensor server. With this setup, we have access to all the messages, users, and channels.

To index this data into a Pinecone vector database, we have a script that reads messages from our database, converts them into vectors using OpenAI, and stores them in your Pinecone account. The script automatically creates Pinecone indexes and namespaces according to the channels on the Bittensor Discord server.

## Essential Parts of the Script

1. **Environment Variables:**
   - The script retrieves the OpenAI and Pinecone API keys from the environment variables.
   - OpenAI API Key: `OPENAI_API_KEY`
   - Pinecone API Key: `PINECONE_API_KEY`

2. **Pinecone and OpenAI Configuration:**
   - The script initializes a Pinecone client using the Pinecone API key.
   - It configures the embedding model from OpenAI using the `text-embedding-3-small` model.

3. **Fetching Users and Channels:**
   - The script includes functions to fetch all users and channels from the database.
   - These functions query the database and store the information in dictionaries for easy access.

4. **Reading Messages:**
   - The script reads messages from the database for a specific channel.
   - It can also filter messages based on a specific point in time (if provided).

5. **Replacing Mentions:**
   - The script replaces user and channel mentions in message content with readable formats.
   - This ensures that the messages are more understandable when converted to vectors.

6. **Parsing Messages to Text Nodes:**
   - Messages are parsed into a format suitable for vector conversion.
   - Each message is converted into a text node containing metadata such as channel name, user details, and timestamps.

7. **Grouping Channels:**
   - The script groups channels based on predefined categories (e.g., discord-subnets, discord-welcome, discord-community).
   - This helps in organizing the messages for indexing.

8. **Creating Indexes:**
   - The script creates indexes in Pinecone for each group of channels.
   - It specifies the dimension and metric for the index (e.g., dimension=1536, metric="cosine").
   - Pinecone namespaces are created according to the channels, and existing namespaces are deleted before new data is indexed.

## Vector Generation and Indexing

1. **Reading and Parsing Messages:**
   - The script reads messages from the database and parses them into text nodes.
   - It handles the replacement of mentions and other formatting to ensure the text is ready for vectorization.

2. **Vector Conversion:**
   - Using the OpenAI embedding model, the script converts the text nodes into vectors.
   - Each message's text is transformed into a vector representation that can be indexed in Pinecone.

3. **Index Creation and Storage:**
   - The script creates indexes in Pinecone with the specified configuration.
   - It stores the generated vectors in the appropriate namespaces within the Pinecone index.
   - The index is then persisted for future use.

# How to use script

1. **Install Dependencies:**
   - Ensure you have all the necessary dependencies installed. You can use the following command to install them:
     ```sh
     pip install -r requirements.txt
     ```

2. **Set Environment Variables:**
   - Set the environment variables for the Discord Messages, OpenAI and Pinecone API keys:
     ```sh
     export DISCORD_MESSAGES_DB_URL='discord-messages-db-url'
     export OPENAI_API_KEY='your-openai-api-key'
     export PINECONE_API_KEY='your-pinecone-api-key'
     ```

3. **Run the Script:**
   - Execute the script by running the following command:
     ```sh
     python3 datura/scripts/discord_messages_indexer.py
     ```

4. **Ignore Index Creation:**
   - If you want to run the script without creating new indexes, use the `--ignore-creating-indexes` flag:
     ```sh
     python3 datura/scripts/discord_messages_indexer.py --ignore-creating-indexes
     ```

5. **Help Command:**
   - To get help or usage information, use the `-h` or `--help` flag:
     ```sh
     python3 datura/scripts/discord_messages_indexer.py --help
     ```
