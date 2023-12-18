import argparse
import time
import subprocess
import template
from template.utils import get_version, send_discord_alert

default_address = "wss://bittensor-finney.api.onfinality.io/public-ws"
webhook_url = ""
current_version = template.__version__

def update_and_restart(pm2_name, wallet_name, wallet_hotkey, address, autoupdate):
    global current_version
    subprocess.run(["pm2", "start", "validators/validator.py", "--interpreter", "python3", "--name", pm2_name, "--", "--wallet.name", wallet_name, "--wallet.hotkey", wallet_hotkey, "--netuid", "18", "--subtensor.network", "local", "--subtensor.chain_endpoint", address])
    while True:
        latest_version = get_version()
        print(f"Current version: {current_version}")
        print(f"Latest version: {latest_version}")

        if current_version != latest_version and latest_version != None:
            if not autoupdate:
                send_discord_alert(f"Your validator not running the latest code ({current_version}). You will quickly lose vturst if you don't update to version {latest_version}", webhook_url)
            print("Updating to the latest version...")
            subprocess.run(["pm2", "delete", pm2_name])
            subprocess.run(["git", "reset", "--hard"])
            subprocess.run(["git", "pull"])
            subprocess.run(["pip", "install", "-e", "."])
            subprocess.run(["pm2", "start", "validators/validator.py", "--interpreter", "python3", "--name", pm2_name, "--", "--wallet.name", wallet_name, "--wallet.hotkey", wallet_hotkey, "--netuid", "18", "--subtensor.network", "local", "--subtensor.chain_endpoint", address])
            current_version = latest_version
        print("All up to date!")
        time.sleep(180)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically update and restart the validator process when a new version is released.",
        epilog="Example usage: python start_validator.py --pm2_name 'net18vali' --wallet_name 'wallet1' --wallet_hotkey 'key123' [--address 'wss://...'] [--no-autoupdate]"
    )

    parser.add_argument("--pm2_name", required=True, help="Name of the PM2 process.")
    parser.add_argument("--wallet_name", required=True, help="Name of the wallet.")
    parser.add_argument("--wallet_hotkey", required=True, help="Hotkey for the wallet.")
    parser.add_argument("--address", default=default_address, help="Subtensor chain_endpoint, defaults to 'wss://bittensor-finney.api.onfinality.io/public-ws' if not provided.")
    parser.add_argument("--no-autoupdate", action='store_false', dest='autoupdate', help="Disable automatic update. Only send a Discord alert. Add your webhook at the top of the script.")

    args = parser.parse_args()

    try:
        update_and_restart(args.pm2_name, args.wallet_name, args.wallet_hotkey, args.address, args.autoupdate)
    except Exception as e:
        parser.error(f"An error occurred: {e}")
