# Vote Token Signer

This script generates a cryptographic signature for a given access key using Bittensor. It will prompt you to provide your access key, wallet name, the location of your wallet, and your wallet password. Note that we do not ask for your password directly; you'll notice a difference in the input prompts: those in CYAN color are what we ask for, while the white ones are prompted by Bittensor.

## Prerequisites
- Python 3.x
- Bittensor library (`bittensor`)
- Bittensor Wallet:
  For more information, please refer to: [Bittensor Wallets Documentation](https://docs.bittensor.com/getting-started/wallets)

## Usage

The script can be executed from the command line with the following syntax:

```sh
python vote_token_signer.py
```

First, it will ask for your access key, which should be generated from our frontend: [vote.datura.ai/auth](https://vote.datura.ai/auth). Next, it will ask for the wallet name you want to use, followed by the location path of that wallet. If you are using the default wallet address, you can simply press enter to continue. Finally, it will ask for your wallet password. As mentioned earlier, we do not ask for or have access to your password.

After providing these inputs, the script will validate the values, sign the access key to generate a signature, validate the signature, and output the JSON-formatted credentials that you can use to authenticate on [vote.datura.ai](https://vote.datura.ai).

---

### Summary of Changes:
- Improved sentence structure and flow for clarity.
- Corrected grammatical errors.
- Enhanced the readability and tone to be more professional.
- Added inline links for better user experience.
