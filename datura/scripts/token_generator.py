import sys
import bittensor as bt


def generate_token(value: str):
    try:
        wallet = bt.wallet(name="coldkey", hotkey="hotkey")
        signature = wallet.hotkey.sign(value.encode()).hex()
        print(f"Signature:\n  {signature}")
    except Exception as e:
        print(f"[!] Error: {e}")


if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) == 1:
        print('\nUsage:\n python token_generator.py "random cahrachters from our frontend"\n')
        sys.exit(1)

    value = sys.argv[1]
    generate_token(value)
