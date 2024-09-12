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
   git clone https://github.com/Datura-ai/smart-scrape.git
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


### Environment Variables Configuration

For setting up the necessary environment variables for your miner or validator, please refer to the [Environment Variables Guide](./docs/env_variables.md).

# Running the Miner

  ```bash
  python -m neurons/miners/miner.py 
      --netuid 22
      --subtensor.network finney
      --wallet.name <your miner wallet>
      --wallet.hotkey <your validator hotkey>
      --axon.port 14000
  ```

# Running the Validator API with Automatic Updates

These validators are designed to run and update themselves automatically. To run a validator, follow these steps:

1. Install this repository, you can do so by following the steps outlined in [the installation section](#installation).
2. Install [Weights and Biases](https://docs.wandb.ai/quickstart) and run `wandb login` within this repository. This will initialize Weights and Biases, enabling you to view KPIs and Metrics on your validator. (Strongly recommended to help the network improve from data sharing)
3. Install [PM2](https://pm2.io/docs/runtime/guide/installation/) and the [`jq` package](https://jqlang.github.io/jq/) on your system.
   **On Linux**:
   ```bash
   sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
   ``` 
   **On Mac OS**
   ```bash
   brew update && brew install jq && brew install npm && sudo npm install pm2 -g && pm2 update
   ```
4. Run the `run.sh` script which will handle running your validator and pulling the latest updates as they are issued. 
   ```bash
   pm2 start run.sh --name smart_scrape_validators_autoupdate -- --wallet.name <your-wallet-name> --wallet.hotkey <your-wallet-hot-key>
   ```

This will run **two** PM2 process: one for the validator which is called `smart_scrape_validators_main_process` by default (you can change this in `run.sh`), and one for the run.sh script (in step 4, we named it `smart_scrape_validators_autoupdate`). The script will check for updates every 30 minutes, if there is an update then it will pull it, install it, restart `smart_scrape_validators_main_process` and then restart itself.

### Detailed Setup Instructions

For step-by-step guidance on setting up and running a miner, validator, or operating on the testnet or mainnet, refer to the following guides:
- [Miner Setup](./docs/running_a_miner.md)
- [Validator Setup](./docs/running_a_validator.md)
- [Testnet Operations](./docs/running_on_testnet.md)
- [Mainnet Operations](./docs/running_on_mainnet.md)
- [Setting Up and Running the Web Application with Validator Integration](./ui/README.md)

---


## Real-time Monitoring with wandb Integration

The text prompting validator sends data to wandb, allowing real-time monitoring with key metrics like:
- Gating model loss
- Hardware usage
- Forward pass time
- Block duration

Data is publicly available at [this link](https://wandb.ai/smart-scrape/smart-scrape-1.0). Note that [data from anonymous users is deleted after 7 days](https://docs.wandb.ai/guides/app/features/anon).

</div>