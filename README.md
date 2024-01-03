
<div align="center">

# **Bittensor Smart-Scrape** <!-- omit in toc -->
<!-- [![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

### The Incentivized Internet <!-- omit in toc -->

<!-- [Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper) -->

</div>


# Introduction

"Smart-Scrape on Bittensor Subnet 22: Simplified AI-Powered Twitter Data Analysis

Smart-Scrape is an efficient tool designed for straightforward Twitter data analysis. Hosted on the Bittensor network, it caters to researchers, marketers, and data analysts. It focuses on simplifying the extraction of valuable insights from Twitter.

Key Features:

- AI-Powered Analysis: Uses AI models to interpret Twitter data, aiming for a clear understanding of user interactions.
- Real-Time Data: Accesses Twitter's database in real-time, ensuring up-to-date information.
- Sentiment Analysis: Identifies emotional tones in tweets, useful for understanding public opinion.
- Metadata Analysis: Examines tweet details like timestamps and retweet counts for a comprehensive data overview.
- Time-Saving for Researchers: Filters relevant data, reducing manual sorting effort.
- Easy-to-Use: Designed to be user-friendly, accommodating both experts and beginners.

Advantages of Smart-Scrape:

- Decentralized: As part of Bittensor, it promises enhanced reliability.
- Customizable: Adapts to specific user needs for targeted analysis.
- Data-Driven Insights: Helps in making informed decisions.
- Versatile: Suitable for various purposes, from market research to academic studies.

Smart-Scrape streamlines Twitter data analysis with its AI capabilities, offering a user-friendly and efficient way to tap into the wealth of information on Twitter.

</div>

---

# Installation
This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.
```bash
git clone https://github.com/surcyf123/smart-scrape.git
cd smart-scrape
python -m pip install -r requirements.txt
python -m pip install -e .
```

</div>

---

Prior to running a miner or validator, you must [create a wallet](https://github.com/opentensor/docs/blob/main/reference/btcli.md) and [register the wallet to a netuid](https://github.com/opentensor/docs/blob/main/subnetworks/registration.md). Once you have done so, you can run the miner and validator with the following commands.
```bash
# To run the miner
python -m neurons/miners/miner.py 
    --netuid 22
    --subtensor.network finney
    --wallet.name <your miner wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port 14000

# To run both the validator & API
python -m neurons/validators/api.py
    --netuid 22
    --subtensor.network finney
    --wallet.name <your validator wallet>  # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode

# To run only the validator
python -m neurons/validators/validator.py
    --netuid 22
    --subtensor.network finney
    --wallet.name <your validator wallet>  # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
```



</div>

---


# Running

These validators are designed to run and update themselves automatically. To run a validator, follow these steps:

1. Install this repository, you can do so by following the steps outlined in [the installation section](#installation).
2. Install [Weights and Biases](https://docs.wandb.ai/quickstart) and run `wandb login` within this repository. This will initialize Weights and Biases, enabling you to view KPIs and Metrics on your validator. (Strongly recommended to help the network improve from data sharing)
3. Install [PM2](https://pm2.io/docs/runtime/guide/installation/) on your system.
   **On Linux**:
   ```bash
   sudo apt update && sudo apt install npm && sudo npm install pm2 -g && pm2 update
   ``` 
   **On Mac OS**
   ```bash
   brew update && brew install npm && sudo npm install pm2 -g && pm2 update
   ```
4. Run the `Miner` script which will handle running your miner.
   ```bash
   pm2 start neurons/miners/miner.py --interpreter /usr/bin/python3 --name miner_1 -- 
    --wallet.name <your-wallet-name> 
    --wallet.hotkey <your-wallet-hot-key> 
    --subtensor.network finney 
    --netuid 22 
    --axon.port <port> 
   ```
5. Run the `Validator & API` script which will handle running your validator.
   ```bash
    pm2 start neurons/validators/api.py --interpreter /usr/bin/python3  --name validator_api -- 
        --wallet.name <your-wallet-name>  
        --netuid 22 
        --wallet.hotkey <your-wallet-hot-key>  
        --subtensor.network finney  
        --logging.debug
   ```

# Real-time monitoring with wandb integration
By default, the text prompting validator sends data to wandb, allowing users to monitor running validators and access key metrics in real-time, such as:
- Gating model loss
- Hardware usage
- Forward pass time
- Block duration

All the data sent to wandb is publicly available to the community at the following [link](https://wandb.ai/smart-scrape/smart-wandb).

You don't need to have a wandb account to access the data or to generate a new run,
but bear in mind that
[data generated by anonymous users will be deleted after 7 days](https://docs.wandb.ai/guides/app/features/anon#:~:text=If%20there's%20no%20account%2C%20we,be%20available%20for%207%20days)
as default wandb policy.
