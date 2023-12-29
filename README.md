
<div align="center">

# **Bittensor Smart-Scrape** <!-- omit in toc -->
<!-- [![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

### The Incentivized Internet <!-- omit in toc -->

<!-- [Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper) -->

</div>


# Introduction

Bittensor Subnet 22's Smart-Scrape: A Technological Vanguard in AI-Powered Twitter Data Analysis

Welcome to the future of data analysis with Smart-Scrape, a groundbreaking innovation in the realm of social media intelligence. Situated on the cutting-edge Bittensor network, Smart-Scrape is not just a tool but a revolution in harnessing the immense power of Twitter's data for insightful, real-time analysis. This platform is meticulously designed for researchers, marketers, data analysts, and anyone in need of extracting deep insights from the bustling world of Twitter.

Core Features of Smart-Scrape:

Advanced AI Integration: Smart-Scrape is at the forefront of AI technology, utilizing sophisticated models to not only scrape Twitter data but to understand and analyze it. This AI-driven approach ensures that the nuances of human communication are not lost, providing a more accurate and profound understanding of Twitter interactions.

Real-Time Data Extraction: Through the use of efficient API keys, Smart-Scrape taps into Twitter's vast database, providing real-time updates. This feature is crucial for staying ahead in a rapidly changing digital landscape, offering users the most current data available.

Deep Sentiment Analysis: By leveraging AI models, Smart-Scrape goes beyond mere data collection. It delves into the heart of tweets, extracting sentiment and emotional undertones. This capability is invaluable for marketers and analysts looking to gauge public opinion or measure brand perception.

Comprehensive Metadata Utilization: Alongside tweet content, Smart-Scrape analyzes tweet metadata, which includes information like timestamps, retweet counts, and user demographics. This holistic approach allows for a more layered understanding of data, providing insights into when and how users interact with content.

Efficient Research Aid: For researchers, Smart-Scrape acts as an indispensable ally. It sifts through the noise, bringing forward relevant information and saving countless hours of manual data sorting.

User-Friendly Interface: Despite its technical prowess, Smart-Scrape boasts an intuitive interface, making it accessible to users of all skill levels. Whether you're a seasoned data scientist or a casual user, Smart-Scrape's design ensures a smooth, user-friendly experience.

Why Choose Smart-Scrape?

Decentralized and Reliable: Being part of the Bittensor network, Smart-Scrape offers a decentralized solution, enhancing reliability and data integrity.

Tailored Analysis: Whether it's a specific question or a broad topic, Smart-Scrape tailors its analysis to your needs, ensuring relevance and precision.

Empowering Decision-Making: The insights provided by Smart-Scrape empower users to make informed decisions, backed by data-driven intelligence.

Versatile Application: From market research to academic studies, Smart-Scrape's applications are as diverse as the needs of its users.

Smart-Scrape is more than a tool; it's a gateway to understanding the pulse of the digital world through Twitter. With its state-of-the-art AI models and robust data extraction capabilities, Smart-Scrape stands as a beacon of innovation in the ever-evolving landscape of data analysis. Join us in embracing this technological marvel and unlock the full potential of Twitter data with Smart-Scrape.

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

# To run the validator
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
   pm2 start run.sh --name text_prompt_validators_autoupdate -- --wallet.name <your-wallet-name> --wallet.hotkey <your-wallet-hot-key>
   ```

This will run **two** PM2 process: one for the validator which is called `smart_scrape_validators_main_process` by default (you can change this in `run.sh`), and one for the run.sh script (in step 4, we named it `text_prompt_validators_autoupdate`). The script will check for updates every 30 minutes, if there is an update then it will pull it, install it, restart `smart_scrape_validators_main_process` and then restart itself.


# Real-time monitoring with wandb integration
By default, the text prompting validator sends data to wandb, allowing users to monitor running validators and access key metrics in real time, such as:
- Gating model loss
- Hardware usage
- Forward pass time
- Block duration

All the data sent to wandb is publicly available to the community at the following [link](https://wandb.ai/opentensor-dev/openvalidators).

You don't need to have a wandb account to access the data or to generate a new run,
but bear in mind that
[data generated by anonymous users will be deleted after 7 days](https://docs.wandb.ai/guides/app/features/anon#:~:text=If%20there's%20no%20account%2C%20we,be%20available%20for%207%20days)
as default wandb policy.

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
