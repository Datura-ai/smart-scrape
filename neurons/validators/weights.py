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

# Utils for weights setting on chain.

import wandb
import torch
import bittensor as bt
import template
   
def init_wandb(self):
    if self.config.wandb_on:
        run_name = f'validator-{self.my_uuid}-{template.__version__}'
        self.config.uid = self.my_uuid
        self.config.hotkey = self.wallet.hotkey.ss58_address
        self.config.run_name = run_name
        self.config.version = template.__version__
        self.config.type = 'validator'

        # Initialize the wandb run for the single project
        run = wandb.init(
            name=run_name,
            project=template.PROJECT_NAME,
            entity=template.ENTITY,
            config=self.config,
            dir=self.config.full_path,
            reinit=True
        )

        # Sign the run to ensure it's from the correct hotkey
        signature = self.wallet.hotkey.sign(run.id.encode()).hex()
        self.config.signature = signature 
        wandb.config.update(self.config, allow_val_change=True)

        bt.logging.success(f"Started wandb run for project '{template.PROJECT_NAME}'")

def set_weights(self, scores):
    # alpha of .3 means that each new score replaces 30% of the weight of the previous weights
    alpha = .3
    if self.moving_average_scores is None:
        self.moving_average_scores = scores.clone()

    # Update the moving average scores
    self.moving_average_scores = alpha * scores + (1 - alpha) * self.moving_average_scores
    bt.logging.info(f"Updated moving average of weights: {self.moving_average_scores}")
    self.subtensor.set_weights(
        netuid=self.config.netuid, 
        wallet=self.wallet, 
        uids=self.metagraph.uids, 
        weights=self.moving_average_scores, 
        wait_for_inclusion=False)
    bt.logging.success("Successfully set weights.")

def update_weights(self, total_scores, steps_passed):
    """ Update weights based on total scores, using min-max normalization for display"""
    avg_scores = total_scores / (steps_passed + 1)

    # Normalize avg_scores to a range of 0 to 1
    min_score = torch.min(avg_scores)
    max_score = torch.max(avg_scores)
    
    if max_score - min_score != 0:
        normalized_scores = (avg_scores - min_score) / (max_score - min_score)
    else:
        normalized_scores = torch.zeros_like(avg_scores)

    bt.logging.info(f"normalized_scores = {normalized_scores}")
    # We can't set weights with normalized scores because that disrupts the weighting assigned to each validator class
    # Weights get normalized anyways in weight_utils
    set_weights(self, avg_scores)