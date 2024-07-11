import sys
import uuid
import bittensor as bt
from requests import request

URL = "http://api-discord.datura.ai/api/get-discord-db-url"


def generate_token(coldkey: str):
    try:
        wallet = bt.wallet(name=coldkey)
        value = uuid.uuid4().hex[:6].upper()
        signature = wallet.coldkey.sign(value.encode()).hex()
        params = {
            "coldkey_address": wallet.coldkey.ss58_address,
            "value": value,
            "signature": signature
        }

        response = request("GET", URL, params=params)

        if response.status_code == 200:
            db_url = response.json().get('url')
            print(f"\n> {db_url}\n")
            print("To use the generated Discord DB URL, please follow the instructions below:\n")
            print(f'\texport DISCORD_MESSAGES_DB_URL="{db_url}"\n')
            print(f'For more details please refer to env docs: https://github.com/Datura-ai/smart-scrape/blob/main/docs/env_variables.md')
        else:
            print(f"[!] Error: Received status code {response.status_code}. Response: {response.text}")
    except Exception as e:
        print(f"[!] Error: {e}")


if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) < 2:
        print('\nUsage:\n python discord_db_url_generator.py [coldkey]\n')
        sys.exit(1)

    coldkey = sys.argv[1]
    generate_token(coldkey)
