
<div align="center">

# **Bittensor Smart-Scrape** <!-- omit in toc -->
<!-- [![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

### The Incentivized Internet <!-- omit in toc -->

<!-- [Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper) -->

</div>


# Introduction

Introducing Bittensor Subnet 22 (Smart-Scrape):: Revolutionizing Data Analysis with Advanced Twitter Integration.

Smart-Scrape is a cutting-edge platform positioned at the vanguard of data extraction and analysis, specifically harnessing the vast and dynamic landscape of Twitter. This solution is expertly crafted to meet the demands of researchers, marketers, and data analysts seeking in-depth insights from social media interactions. Operating on the decentralized Bittensor network, Smart-Scrape ensures a seamless, reliable, and high-quality extraction of text data via API, fostering an ecosystem of transparent and unbiased intelligence mining and response generation.

Our platform marks a significant advancement in how we approach Twitter data analysis. By honing in on the essence of user queries and topics, Smart-Scrape meticulously sifts through Twitter's extensive database. This process involves:

- **Question or Topic Analysis:** Initially, we dissect your inquiry or area of interest to grasp its core elements fully.

- **Twitter Data Search:** We then probe into Twitter, seeking relevant conversations, trends, and perspectives that align with your prompt.

- **Synthesis and Response:** Our advanced algorithms analyze this data, synthesizing the findings into a comprehensive response.

- **Output:** You receive a concise, insightful introduction derived from our analysis, without any additional commentary.

Smart-Scrape stands out by integrating state-of-the-art AI models, similar to those used in creating Microsoft's WizardLM, to refine and enhance the data analysis process. This involves a unique method of generating synthetic prompt-response pairs, archived in wandb [wandb.ai/smart-scrape/twitter-data](https://wandb.ai/smart-scrape/twitter-data). We recycle model outputs back into our system, employing a strategy of prompt evolution and data augmentation. This not only enables the distillation of advanced AI models into more efficient forms but also mirrors the high performance of their larger counterparts.

The platform's ability to utilize synthetic data effectively overcomes the challenges typically associated with data collection and curation, expediting the development of robust and flexible AI models. With Smart-Scrape, you're not just accessing data; you're tapping into a rich reservoir of intelligence, paving the way for AI models that reflect the intricate understanding and response capabilities of their forebears.

Join the journey with Smart-Scrape on Bittensor subnet 41, your portal to unparalleled Twitter data analysis, and be part of the transformative wave in AI-driven data intelligence. Embrace the future with Smart-Scrape – Where Data Meets Intelligence!"

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
    --netuid 41  
    --subtensor.network test 
    --wallet.name <your miner wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port 14000

# To run the validator
python -m neurons/validators/validator.py
    --netuid 41
    --subtensor.network test 
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
