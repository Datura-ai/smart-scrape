# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright d© 2023 Opentensor Foundation

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
import datura
import multiprocessing
import time

import torch
from multiprocessing import Queue


def init_wandb(self):
    try:
        if self.config.wandb_on:
            run_name = f"validator-{self.uid}-{datura.__version__}"
            self.config.uid = self.uid
            self.config.hotkey = self.wallet.hotkey.ss58_address
            self.config.run_name = run_name
            self.config.version = datura.__version__
            self.config.type = "validator"

            # Initialize the wandb run for the single project
            run = wandb.init(
                name=run_name,
                project=datura.PROJECT_NAME,
                entity=datura.ENTITY,
                config=self.config,
                dir=self.config.full_path,
                reinit=True,
            )

            # Sign the run to ensure it's from the correct hotkey
            signature = self.wallet.hotkey.sign(run.id.encode()).hex()
            self.config.signature = signature
            wandb.config.update(self.config, allow_val_change=True)

            bt.logging.success(f"Started wandb run for project '{datura.PROJECT_NAME}'")
    except Exception as e:
        bt.logging.error(f"Error in init_wandb: {e}")
        raise


class RetryException(Exception):
    pass


def on_retry(exception, tries_remaining, delay):
    attempt = 6 - tries_remaining  # Assuming 5 total tries
    bt.logging.info(f"Retry attempt {attempt}, will retry in {delay} seconds...")


def set_weights_subtensor(queue, wallet, netuid, uids, weights, config, version_key):
    try:
        subtensor = bt.subtensor(config=config)
        success, message = subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=False,
            wait_for_finalization=False,
            version_key=version_key,
        )

        # Send the success status back to the main process
        queue.put(success)
        return success, message
    except Exception as e:
        bt.logging.error(f"Failed to set weights on chain with exception: { e }")
        return False, message


def set_weights_with_retry(self, processed_weight_uids, processed_weights):
    max_retries = 9  # Maximum number of retries
    retry_delay = 45  # Delay between retries in seconds
    ttl = 200  # Time-to-live for each process attempt in seconds

    bt.logging.info("Initiating weight setting process on Bittensor network.")
    for attempt in range(max_retries):
        success = False
        queue = Queue()  # Create a new queue for each attempt
        process = multiprocessing.Process(
            target=set_weights_subtensor,
            args=(
                queue,  # Pass the queue as the first argument
                self.wallet,
                self.config.netuid,
                processed_weight_uids,
                processed_weights,
                self.config,
                datura.__weights_version__,
            ),
        )
        process.start()
        process.join(timeout=ttl)

        if not process.is_alive():
            process.terminate()  # Ensure the process is terminated
            process.join()  # Clean up the terminated process

            # Check the queue for the success status
            if not queue.empty():
                queue_success = queue.get()
                # Directly handle the return value without unpacking
                if isinstance(queue_success, tuple):
                    # If it's a tuple, unpack it
                    success_status, message = queue_success
                    success = success_status
                    if success_status:
                        bt.logging.success(
                            f"Set Weights Completed set weights action successfully. Message: '{message}'"
                        )
                    else:
                        bt.logging.info(
                            f"Set Weights Attempt failed with message: '{message}', retrying in {retry_delay} seconds..."
                        )
                else:
                    # Handle the case where the return value is not a tuple (e.g., a boolean)
                    success = queue_success
                    if success:
                        bt.logging.success(
                            f"Set Weights Completed set weights action successfully, Response: {success}"
                        )
                    else:
                        bt.logging.info(
                            f"Set Weights Attempt failed. retrying in {retry_delay} seconds..., Response: {success}"
                        )
            else:
                bt.logging.info(
                    f"Set Weights Attempt {attempt + 1} failed, no response received, retrying in {retry_delay} seconds..."
                )
        else:
            process.terminate()  # Ensure the process is terminated before retrying
            process.join()  # Clean up the terminated process
            bt.logging.info(
                f"Set Weights Attempt {attempt + 1} failed, process did not complete in time, retrying in {retry_delay} seconds.."
            )
        if not success:
            time.sleep(retry_delay)  # Wait for the specified delay before retrying
        else:
            break  # Exit the retry loop on success
    if success:
        bt.logging.success(
            f"Final Result: Successfully set weights after {attempt + 1} attempts."
        )
    else:
        bt.logging.error(
            f"Final Result: Failed to set weights after {attempt + 1} attempts."
        )

    return success


def get_weights(self):
    if torch.all(self.moving_averaged_scores == 0):
        bt.logging.info("All moving averaged scores are zero, skipping weight setting.")
        return {}

    raw_weights = torch.nn.functional.normalize(self.moving_averaged_scores, p=1, dim=0)

    (
        processed_weight_uids,
        processed_weights,
    ) = bt.utils.weight_utils.process_weights_for_netuid(
        uids=self.metagraph.uids.to("cpu"),
        weights=raw_weights.to("cpu"),
        netuid=self.config.netuid,
        subtensor=self.subtensor,
        metagraph=self.metagraph,
    )

    weights_dict = {
        str(uid.item()): weight.item()
        for uid, weight in zip(processed_weight_uids, processed_weights)
    }

    return weights_dict


def set_weights(self):
    if torch.all(self.moving_averaged_scores == 0):
        bt.logging.info("All moving averaged scores are zero, skipping weight setting.")
        return
    # Calculate the average reward for each uid across non-zero values.
    # Replace any NaN values with 0.
    raw_weights = torch.nn.functional.normalize(self.moving_averaged_scores, p=1, dim=0)
    bt.logging.trace("raw_weights", raw_weights)
    bt.logging.trace("top10 values", raw_weights.sort()[0])
    bt.logging.trace("top10 uids", raw_weights.sort()[1])

    # Process the raw weights to final_weights via subtensor limitations.
    (
        processed_weight_uids,
        processed_weights,
    ) = bt.utils.weight_utils.process_weights_for_netuid(
        uids=self.metagraph.uids.to("cpu"),
        weights=raw_weights.to("cpu"),
        netuid=self.config.netuid,
        subtensor=self.subtensor,
        metagraph=self.metagraph,
    )

    weights_dict = {
        str(uid.item()): weight.item()
        for uid, weight in zip(processed_weight_uids, processed_weights)
    }

    # Log the weights dictionary
    bt.logging.info(f"Attempting to set weights action for {weights_dict}")

    bt.logging.info(
        f"Attempting to set weights details begins: ================ for {len(processed_weight_uids)} UIDs"
    )
    uids_weights = [
        f"UID - {uid.item()} = Weight - {weight.item()}"
        for uid, weight in zip(processed_weight_uids, processed_weights)
    ]
    for i in range(0, len(uids_weights), 4):
        bt.logging.info(" | ".join(uids_weights[i : i + 4]))
    bt.logging.info(f"Attempting to set weights details ends: ================")

    # Call the new method to handle the process with retry logic
    success = set_weights_with_retry(self, processed_weight_uids, processed_weights)
    return success


def update_weights(self, total_scores, steps_passed):
    try:
        """Update weights based on total scores, using min-max normalization for display"""
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
    except Exception as e:
        bt.logging.error(f"Error in update_weights: {e}")
        raise
