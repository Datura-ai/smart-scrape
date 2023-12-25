<div align="left">

# **Cortex.t Subnet** <!-- omit in toc -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
---

---
- [Introduction](#introduction)
- [Setup](#setup)
- [Mining](#mining)
- [Validating](#validating)
- [License](#license)


## Introduction

**IMPORTANT**: If you are new to Bittensor, please checkout the [Bittensor Website](https://bittensor.com/) before proceeding to the [Setup](#setup) section. 

Introducing Bittensor Subnet 18 (Smart-Scrape):: Revolutionizing Data Analysis with Advanced Twitter Integration.

Smart-Scrape is a cutting-edge platform positioned at the vanguard of data extraction and analysis, specifically harnessing the vast and dynamic landscape of Twitter. This solution is expertly crafted to meet the demands of researchers, marketers, and data analysts seeking in-depth insights from social media interactions. Operating on the decentralized Bittensor network, Smart-Scrape ensures a seamless, reliable, and high-quality extraction of text data via API, fostering an ecosystem of transparent and unbiased intelligence mining and response generation.

Our platform marks a significant advancement in how we approach Twitter data analysis. By honing in on the essence of user queries and topics, Smart-Scrape meticulously sifts through Twitter's extensive database. This process involves:

- **Question or Topic Analysis:** Initially, we dissect your inquiry or area of interest to grasp its core elements fully.

- **Twitter Data Search:** We then probe into Twitter, seeking relevant conversations, trends, and perspectives that align with your prompt.

- **Synthesis and Response:** Our advanced algorithms analyze this data, synthesizing the findings into a comprehensive response.

- **Output:** You receive a concise, insightful introduction derived from our analysis, without any additional commentary.

Smart-Scrape stands out by integrating state-of-the-art AI models, similar to those used in creating Microsoft's WizardLM, to refine and enhance the data analysis process. This involves a unique method of generating synthetic prompt-response pairs, archived in wandb [wandb.ai/smart-scrape/twitter-data](https://wandb.ai/smart-scrape/twitter-data). We recycle model outputs back into our system, employing a strategy of prompt evolution and data augmentation. This not only enables the distillation of advanced AI models into more efficient forms but also mirrors the high performance of their larger counterparts.

The platform's ability to utilize synthetic data effectively overcomes the challenges typically associated with data collection and curation, expediting the development of robust and flexible AI models. With Smart-Scrape, you're not just accessing data; you're tapping into a rich reservoir of intelligence, paving the way for AI models that reflect the intricate understanding and response capabilities of their forebears.

Join the journey with Smart-Scrape on Bittensor subnet 41, your portal to unparalleled Twitter data analysis, and be part of the transformative wave in AI-driven data intelligence. Embrace the future with Smart-Scrape – Where Data Meets Intelligence!"

---

This text aligns with the style and content of the original while incorporating the unique aspects of your project, Smart-Scrape.


## Setup



### Before you proceed
Before you proceed with the installation of the subnet, note the following: 

**IMPORTANT**: We **strongly recommend** before proceeding that you test both subtensor and OpenAI API keys. Ensure you are running Subtensor locally to minimize chances of outages and improve the latency/connection. 

After exporting your OpenAI API key to your bash profile, test the streaming service for both the gpt-3.5-turbo and gpt-4 engines using ```./neurons/test_openai.py```. Neither the miner or the validator will function without a valid and working [OpenAI API key](https://platform.openai.com/). 

**IMPORTANT:** Make sure you are aware of the minimum compute requirements for cortex.t. See the [Minimum compute YAML configuration](./min_compute.yml).
Note that this subnet requires very little compute. The main functionality is api calls, so we outsource the compute to openai. The cost for mining and validating on this subnet comes from api calls, not from compute. Please be aware of your API costs and monitor accordingly.

A high tier key is required for both mining and validations so it is important if you do not have one to work your way up slowly by running a single miner or small numbers of miners whilst payiing attention to your usage and limits.


### Installation

Download the repository, navigate to the folder and then install the necessary requirements with the following chained command.

```git clone git@github.com:surcyf123/smart-scrape.git && cd smart-scrape && pip install -e .```

Prior to proceeding, ensure you have a registered hotkey on subnet 18 mainnet. If not, run the command `btcli s register --netuid 18 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey]`.

In order to run a miner or validator you must first set your OpenAI key to your profile with the following command.

```echo "export OPENAI_API_KEY=your_api_key_here">>~/.bashrc && source ~/.bashrc```

```echo "export TWITTER_BEARER_TKEN=your_api_key_here">>~/.bashrc && source ~/.bashrc```


## Mining

You can run miner: 

`python3 miner/miner.py --wallet.name miner --wallet.hotkey <WALLET NAME>  --subtensor.network test --netuid 41 --axon.port 14000`

Or you can launch your miners via pm2 using the following command. 

`pm2 start ./neurons/miner.py --interpreter python3 -- --netuid 18 --subtensor.network <LOCAL/FINNEY> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME> --axon.port <PORT>`


## Validating

You can run twitter validator:
`python3 validators/validator.py --wallet.name validator --wandb.off --netuid 41 --wallet.hotkey <WALLET NAME> --subtensor.network test`

Or you can launch your validator via pm2 using the following command.

`pm2 start ./validators/validator.py --interpreter python3 -- --netuid 18 --subtensor.network <LOCAL/FINNEY> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>`


## Logging

As cortex.t supports streaming natively, you do not (and should not) enable `logging.trace` or `logging.debug` as all of the important information is already output to `logging.info` which is set as default.

---

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
