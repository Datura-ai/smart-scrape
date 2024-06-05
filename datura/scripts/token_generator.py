import sys
import bittensor as bt


def generate_token(value: str, coldkey: str, hotkey: str):
    try:
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)
        signature = wallet.hotkey.sign(value.encode()).hex()
        print(f"\nSignature:\n  {signature}")
        print(f"\nss58 address:\n  {wallet.hotkey.ss58_address}\n")
    except Exception as e:
        print(f"[!] Error: {e}")


if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) < 4:
        args = '"random cahrachters from our frontend" [coldkey] [hotkey]'
        print(f'\nUsage:\n python token_generator.py {args}\n')
        sys.exit(1)

    value = sys.argv[1]
    coldkey = sys.argv[2]
    hotkey = sys.argv[3]
    generate_token(value, coldkey, hotkey)
