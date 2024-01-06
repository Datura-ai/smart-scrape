<div align="center">

# **Bittensor Smart-Scrape**
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>

## Introduction

**Bittensor Smart-Scrape:** Streamlining Twitter Data Analysis on Subnet 22

Welcome to Smart-Scrape, a cutting-edge tool hosted on the Bittensor network, designed for effective and simplified analysis of Twitter data. This tool is ideal for researchers, marketers, and data analysts who seek to extract insightful information from Twitter with ease.

### Key Features

- **AI-Powered Analysis:** Harnesses artificial intelligence to delve into Twitter data, providing deeper insights into user interactions.
- **Real-Time Data Access:** Connects directly with Twitter's database for the latest information.
- **Sentiment Analysis:** Determines the emotional tone of tweets, aiding in understanding public sentiment.
- **Metadata Analysis:** Dives into tweet details like timestamps and retweet counts for a comprehensive view.
- **Time-Efficient:** Minimizes manual data sorting, saving valuable research time.
- **User-Friendly Design:** Suitable for both beginners and experts.

### Advantages

- **Decentralized Platform:** Ensures reliability through its placement on the Bittensor network.
- **Customizability:** Tailors data analysis to meet specific user requirements.
- **Informed Decision-Making:** Facilitates data-driven strategies.
- **Versatility:** Applicable for diverse research fields, from market analysis to academic studies.

---

## Installation

**Requirements:** Python 3.8 or higher

1. Clone the repository:
   ```bash
   git clone https://github.com/surcyf123/smart-scrape.git
   ```
2. Install the requirements:
   ```bash
   cd smart-scrape
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```

---

## Preparing Your Environment

Before running a miner or validator, ensure to:

- [Create a wallet](https://github.com/opentensor/docs/blob/main/reference/btcli.md).
- [Register the wallet to a netuid](https://github.com/opentensor/docs/blob/main/subnetworks/registration.md).

### Running Commands

- **To run the miner:**
  ```bash
  python -m neurons/miners/miner.py 
      --netuid 22
      --subtensor.network finney
      --wallet.name <your miner wallet>
      --wallet.hotkey <your validator hotkey>
      --logging.debug
      --axon.port 14000
  ```

- **To run both the validator & API:**
  ```bash
  python -m neurons/validators/api.py
      --netuid 22
      --subtensor.network finney
      --wallet.name <your validator wallet>
      --wallet.hotkey <your validator hotkey>
      --logging.debug
  ```

- **To run only the validator:**
  ```bash
  python -m neurons/validators/validator.py
      --netuid 22
      --subtensor.network finney
      --wallet.name <your validator wallet>
      --wallet.hotkey <your validator hotkey>
      --logging.debug
  ```

### Detailed Setup Instructions

For step-by-step guidance on setting up and running a miner, validator, or operating on the testnet or mainnet, refer to the following guides:
- [Miner Setup](./docs/running_a_miner.md)
- [Validator Setup](./docs/running_a_validator.md)
- [Testnet Operations](./docs/running_on_testnet.md)
- [Mainnet Operations](./docs/running_on_mainnet.md)

---

## Environment Variables Configuration

For setting up the necessary environment variables for your miner or validator, please refer to the [Environment Variables Guide](./docs/env_variables.md).

---

## Running Your Validators

Validators are designed to run and update themselves automatically. Follow these steps to run a validator:

1. Install this repository as outlined in [the installation section](#installation).
2. Set up [Weights and Biases](https://docs.wandb.ai/quickstart) and run `wandb login` within this repository for KPIs and Metrics monitoring.
3. Install [PM2](https://pm2.io/docs/runtime/guide/installation/).
   - **On Linux**:
     ```bash
     sudo apt update && sudo apt install npm && sudo npm install pm2 -g && pm2 update
     ```
   - **On Mac OS**:
     ```bash
     brew update && brew install npm && sudo npm install pm2 -g && pm2 update
     ```
4. Run the `Miner` script:
   ```bash
   pm2 start neurons/miners/miner.py --interpreter /usr/bin/python3 --name miner_1 --
    --wallet.name <your-wallet-name> 
    --wallet.hotkey <your-wallet-hot-key> 
    --subtensor.network <network> 
    --netuid 22 
    --axon.port <port> 
   ```
5. Run the `Validator & API` script:
   ```bash
    pm2 start neurons/validators/api.py --interpreter /usr/bin/python3  --name validator_api --
        --wallet.name <your-wallet-name>  
        --netuid 22 
        --wallet.hotkey <your-wallet-hot-key>  
        --subtensor.network <network>  
        --logging.debug
   ```

---

## Real-time Monitoring with wandb Integration

The text prompting validator sends data to wandb, allowing real-time monitoring with key metrics like:
- Gating model loss
- Hardware usage
- Forward pass time
- Block duration

Data is publicly available at [this link](https://wandb.ai/smart-scrape/smart-wandb). Note that [data from anonymous users is deleted after 7 days](https://docs.wandb.ai/guides/app/features/anon).

</div>