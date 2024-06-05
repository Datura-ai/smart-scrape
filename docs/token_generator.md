# Token Generator Documentation

This script generates a cryptographic signature for a given string using the Bittensor.
It two three command-line arguments: a access token value (generated from frontend) to be signed, and a coldkey.

## Prerequisites
- Python 3.x
- Bittensor library (`bittensor`)
- Bittensor Wallet:
  for more information please refer to: https://docs.bittensor.com/getting-started/wallets

## Usage

The script can be executed from the command line with the following syntax:

```sh
python token_generator.py <access_key> <coldkey>
```

### Arguments

- `<value>`: A string of random characters that you want to sign.
- `<coldkey>`: The name of the coldkey associated with the wallet.

### Example

```sh
python token_generator.py access_key coldkey_name
```
