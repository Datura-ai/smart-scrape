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

Introducing Bittensor Subnet 18 (Cortex.t): A Pioneering Platform for AI Development and Synthetic Data Generation.

Cortex.t stands at the forefront of artificial intelligence, offering a dual-purpose solution that caters to the needs of app developers and innovators in the AI space. This platform is meticulously designed to deliver reliable, high-quality text and image responses through API usage, utilising the decentralised Bittensor network. It serves as a cornerstone for creating a fair, transparent, and manipulation-free environment for the incentivised production of intelligence (mining) and generation and fulfilment of diverse user prompts.

Our initiative is a leap forward in redefining the reward system for text and image prompting with a commitment to providing stability and reassurance to developers. By focusing on the value delivered to clients, we alleviate the concerns of data inconsistencies that often plague app development. The quality of Cortex.t is seamlessly integrated within the Bittensor network, allowing developers to harness the power of multiple subnets and modalities by building directly onto an existing validator, or through an API key from [Corcel](https://corcel.io).

Cortex.t is also a transformative platform leveraging advanced AI models to generate synthetic prompt-response pairs. This novel method yields a comprehensive dataset of interactions, archived in wandb [wandb.ai/cortex-t/synthetic-QA](https://wandb.ai/cortex-t/synthetic-QA). The process involves recycling model outputs back into the system, using a prompt evolution and data augmentation strategy similar to Microsoft's approach in developing WizardLM. This enables the distillation of sophisticated AI models into smaller, yet efficient counterparts, mirroring the performance of their larger predecessors. Ultimately, Cortex.t democratizes access to high-end AI technology, encouraging innovation and customization.

By leveraging synthetic data, Cortex.t circumvents the traditional challenges of data collection and curation, accelerating the development of AI models that are both robust and adaptable. This platform is your gateway to AI mastery, offering the unique opportunity to train your models with data that reflects the depth and versatility of the parent model. With SynthPairPro, you're not just collecting data; you're capturing intelligence, providing a path to creating AI models that mirror the advanced understanding and response capabilities of their predecessors.

Join us at Cortex.t, your bridge to AI excellence, and democratise access to top-level AI capabilities. Be part of the AI revolution and stay at the forefront of innovation with SynthPairPro – Synthesizing Intelligence, Empowering the Future!


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

```git clone https://github.com/BitAPAI/cortex.t.git && cd cortex.t && pip install -e .```

Prior to proceeding, ensure you have a registered hotkey on subnet 18 mainnet. If not, run the command `btcli s register --netuid 18 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey]`.

In order to run a miner or validator you must first set your OpenAI key to your profile with the following command.

```echo "export OPENAI_API_KEY=your_api_key_here">>~/.bashrc && source ~/.bashrc```


## Mining

You can launch your miners via pm2 using the following command. 

`pm2 start ./neurons/miner.py --interpreter python3 -- --netuid 18 --subtensor.network <LOCAL/FINNEY> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME> --axon.port <PORT>`


## Validating

You can launch your validator via pm2 using the following command.

`pm2 start ./neurons/validator.py --interpreter python3 -- --netuid 18 --subtensor.network <LOCAL/FINNEY> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>`


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
