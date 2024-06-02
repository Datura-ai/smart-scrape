# Token Generator Documentation

This script generates a cryptographic signature and a JWT (JSON Web Token) for a given secret string using the Bittensor library. It takes three command-line arguments: a secret value, a coldkey name, and a hotkey name. The script uses the specified coldkey and hotkey to create a wallet, then signs the secret value with the hotkey, generates a JWT, and prints both the resulting signature and the JWT. This can be used for secure authentication or verification purposes.

## Prerequisites
- Python 3.x
- Bittensor library (`bittensor`)
- PyJWT library (`jwt`)
- Bittensor Wallet:
  For more information please refer to: [Bittensor Wallet Documentation](https://docs.bittensor.com/getting-started/wallets)

## Usage

The script can be executed from the command line with the following syntax:

```sh
python token_generator.py <secret> <coldkey> <hotkey>
```

### Arguments

- `<secret>`: A string of random characters that you want to sign and include in the JWT.
- `<coldkey>`: The name of the coldkey associated with the wallet.
- `<hotkey>`: The name of the hotkey associated with the wallet.

### Example

```sh
python token_generator.py "random characters from our frontend" coldkey_name hotkey_name
```

### Output

The script will print the following:
- The secret value.
- The generated JWT.
- The cryptographic signature of the secret value using the hotkey.

### Error Handling

If an error occurs during the execution, the script will print an error message indicating the nature of the error.

For more detailed information, you can refer to the documentation of the libraries used in the script:
- [Bittensor Documentation](https://docs.bittensor.com/)
- [PyJWT Documentation](https://pyjwt.readthedocs.io/en/latest/)
