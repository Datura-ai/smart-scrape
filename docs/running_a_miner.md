# Bittensor (Smart-Scrape) Miner Setup Guide

This guide provides detailed instructions for setting up and running a Bittensor Smart-Scrape miner using the smart-scrape repository.

## Prerequisites
Before you begin, ensure that you have PM2 installed to manage your processes. If you don’t have it installed, follow the installation guide [here](https://pm2.io/docs/runtime/guide/installation/).

## 1. Install the smart-scrape Repository
To start, install the smart-scrape repository. Navigate to the directory where you’ve cloned or downloaded `smart-scrape`, and run the following command:

```sh
python -m pip install -e ~/smart-scrape
```

## 2. Install Specific Miner Requirements
If there are any additional requirements for your miner, install them by running:

```sh
python -m pip install -r requirements.txt
```

## 3. Load and Run the Miner
Once you have installed the necessary packages, you can load and run the miner using PM2. Set the `CUDA_VISIBLE_DEVICES` variable to the GPU you want to use, and adjust the other variables according to your setup.

```sh
CUDA_VISIBLE_DEVICES=0 pm2 start neurons/miners/miner.py \
--miner.name Smart-Scrape \
--interpreter <path-to-python-binary> -- \
--wallet.name <wallet-name> \
--wallet.hotkey <wallet-hotkey> \
--netuid <netuid> \
--subtensor.network <network> \
--logging.debug \
--axon.port <port>

#Example
pm2 start neurons/miners/miner.py --interpreter /usr/bin/python3 --name miner_1 -- --wallet.name miner --wallet.hotkey default --subtensor.network testnet --netuid 41 --axon.port 14001  --logging.debug

```

### Variable Explanation
- `--wallet.name`: Provide the name of your wallet.
- `--wallet.hotkey`: Enter your wallet's hotkey.
- `--netuid`: Use `41` for testnet.
- `--subtensor.network`: Specify the network you want to use (`finney`, `test`, `local`, etc).
- `--logging.debug`: Adjust the logging level according to your preference.
- `--axon.port`: Specify the port number you want to use.

- `--miner.name`: Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name 
- `--miner.mock_dataset`: If True, the miner will retrieve data from mock dataset
- `--miner.blocks_per_epoch`: Blocks until the miner sets weights on chain
- `--miner.no_set_weights`: If True, the miner does not set weights.

## Conclusion
By following this guide, you should be able to setup and run a Smart-scrape miner using the smart-scrape repository with PM2. Ensure that you monitor your processes and check the logs regularly for any issues or important information. For more details or if you encounter any problems, refer to the official documentation or seek help from the community.
