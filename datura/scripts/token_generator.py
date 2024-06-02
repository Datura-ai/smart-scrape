import sys
import jwt
import bittensor as bt
import datetime


def generate_token(secret: str, coldkey: str, hotkey: str):
    try:
        jwt_secret = create_jwt(secret)
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)
        signature = wallet.hotkey.sign(secret.encode()).hex()
        print(f"\nSecret:\n  {secret}")
        print(f"\nAccess Token:\n  {jwt_secret}")
        print(f"\nSignature:\n  {signature}\n")
    except Exception as e:
        print(f"[!] Error: {e}")


def create_jwt(secret: str):
    payload = {
        "data": secret,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    return token


if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) < 4:
        args = '[jwt_secret] [coldkey] [hotkey]'
        print(f'\nUsage:\n python token_generator.py {args}\n')
        sys.exit(1)

    secret = sys.argv[1]
    coldkey = sys.argv[2]
    hotkey = sys.argv[3]
    generate_token(secret, coldkey, hotkey)
