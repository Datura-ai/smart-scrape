# Token Generator Documentation

This script generates a cryptographic signature for a given string using the Bittensor. It takes three command-line arguments: a value to be signed, a coldkey, and a hotkey. The script uses the specified coldkey and hotkey to create a wallet, then signs the value with the hotkey, and prints the resulting signature. This can be used for secure authentication or verification purposes.

## Prerequisites
- Python 3.x
- Bittensor library (`bittensor`)
- Bittensor Wallet:
  for more information please refer to: https://docs.bittensor.com/getting-started/wallets

## Usage

The script can be executed from the command line with the following syntax:

```sh
python token_generator.py <value> <coldkey> <hotkey>
```

### Arguments

- `<value>`: A string of random characters that you want to sign.
- `<coldkey>`: The name of the coldkey associated with the wallet.
- `<hotkey>`: The name of the hotkey associated with the wallet.

### Example

```sh
python token_generator.py "random characters from our frontend" coldkey_name hotkey_name
```
