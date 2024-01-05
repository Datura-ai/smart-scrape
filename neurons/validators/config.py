# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

import os
import torch
import argparse
import bittensor as bt
from loguru import logger
from reward import DefaultRewardFrameworkConfig


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)
    # bt.wallet.check_config(config)
    # bt.subtensor.check_config(config)

    if config.mock:
        config.neuron.mock_dataset = False
        config.wallet._mock = True

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            # config.neuron.name,
            'validator'
        )
    )
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        logger.level("EVENTS", no=38, icon="📝")
        logger.add(
            config.neuron.full_path + "/" + "completions.log",
            rotation=config.neuron.events_retention_size,
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            level="EVENTS",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )

def add_args(cls, parser):
    parser.add_argument(
        "--netuid", type=int, help="Prompting network netuid", default=1
    )

    parser.add_argument('--wandb.off', action='store_false', dest='wandb_on')

    parser.set_defaults(wandb_on=True)

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name. ",
        default="core_smart_scrape_validator",
    )

    # Netuid Arg
    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run the validator on.",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    parser.add_argument(
        "--neuron.disable_log_rewards",
        action="store_true",
        help="Disable all reward logging, suppresses reward functions and their values from being logged to wandb.",
        default=False,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter, how much to add of the new observation.",
        default=0.05,
    )

    parser.add_argument(
        "--reward.dpo_weight",
        type=float,
        help="Weight for the dpo reward model",
        default=DefaultRewardFrameworkConfig.dpo_model_weight,
    )

    parser.add_argument(
        "--reward.rlhf_weight",
        type=float,
        help="Weight for the rlhf reward model",
        default=DefaultRewardFrameworkConfig.rlhf_model_weight,
    )

    parser.add_argument(
        "--reward.prompt_based_weight",
        type=float,
        help="Weight for the prompt-based reward model",
        default=DefaultRewardFrameworkConfig.prompt_model_weight,
    )

    parser.add_argument(
        "--neuron.run_random_miner_syn_qs_interval",
        type=int,
        help="Sets the interval, in seconds, for querying a random subset of miners with synthetic questions. Set to a positive value to enable. A value of 0 disables this feature.",
        default=0,
    )
    
    parser.add_argument(
        "--neuron.run_all_miner_syn_qs_interval",
        type=int,
        help="Sets the interval, in seconds, for querying all miners with synthetic questions. Set to a positive value to enable. A value of 0 disables this feature.",
        default=0,
    )



def config(cls):
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser) 
    return bt.config(parser)
