# Bittensor Validator Setup Guide

This document outlines the steps to set up and run a Bittensor node using the smart-scrape repository on the testnet. You can mirror the same but change `--subtensor.network` to `finney`, `local`, or your own endpoint with `--subtensor.chain_endpoint <ENDPOINT>`. Follow the instructions below to set up your environment, install necessary packages, and start the Bittensor process.

We recommend using `pm2` to manage your processes. See the pm2 install [guide](https://pm2.io/docs/runtime/guide/installation/) for more info.

## Hardware requirements:
- Recommended: A100 80GB
- Minimum: A40 48GB or A6000 48GB

## 0. Install Conda Environment
Create a new conda environment named `val` with Python 3.10.

```sh
conda create -n val python=3.10
```

Activate the conda environment:

```sh
conda activate val
```

## 1. Install Bittensor

Install the Bittensor paåckage directly from the GitHub repository on the `revolution` branch.

```sh
python -m pip install git+https://github.com/opentensor/bittensor.git@revolution
```

## 2. Clone the smart-scrape repository
Clone the smart-scrape repository and install the package in editable mode.

```sh
git clone https://github.com/surcyf123/smart-scrape.git
cd smart-scrape
python -m pip install -e .
```

## 3. Set up Your Wallet
Create new cold and hot keys for your wallet:

```sh
btcli wallet new_coldkey
btcli wallet new_hotkey
```

### 3.1 Get some TAO
Use the faucet command to get some TAO for your wallet on the test network (or get real Tao on mainnet by purchasing OTC or mining yourself):

```sh
btcli wallet faucet --wallet.name validator --subtensor.network test
```

## 4. Register your UID on the Network
Register your UID on the test network:

```sh
btcli subnets register --subtensor.network test
```

## 5. Start the Process
Check which GPUs are available by running:

```sh
nvidia-smi
```

Launch the process using `pm2` and specify the GPU to use by setting the `CUDA_VISIBLE_DEVICES` variable. Adjust the following command to your local paths, available GPUs, and other preferences:

```sh
CUDA_VISIBLE_DEVICES=1 pm2 start ~/tutorial/smart-scrape/neurons/validators/validator.py \
    --name validator1 --interpreter ~/miniconda3/envs/val/bin/python -- \
    --wallet.name validator --netuid 8 --wallet.hotkey vali --subtensor.network test \
    --logging.debug --neuron.reward_path ~/.bittensor/test-subnet1-validators \
    --axon.port 8899
```

## 6. Monitor Your Process
Use the following `pm2` commands to monitor the status and logs of your process:

```sh
pm2 status
pm2 logs 0
```

# Conclusion
By following the steps above, you should have successfully set up and started a Bittensor node using the smart-scrape repository. Make sure to monitor your process regularly and ensure that it's running smoothly. If you encounter any issues or have any questions, refer to the [Bittensor documentation](https://github.com/opentensor/smart-scrape/docs/) or seek help from the community.


> Note: Make sure you have at least >50GB free disk space for wandb logs.