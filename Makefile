# Define your workspace folder here.
WORKSPACE_FOLDER := /Users/theiskaa/dev/projects/smart-scrape

# Miner command
MINER_CMD := python $(WORKSPACE_FOLDER)/neurons/miners/miner.py --wallet.name miner --wallet.hotkey default --subtensor.network test --netuid 41 --axon.port 14000 --miner.mock_dataset False --miner.intro_text True --miner.save_logs True

# API command
# Note: Do not forget to update only_allowed_miners with your miner (at the end of the file).
API_CMD := python $(WORKSPACE_FOLDER)/neurons/validators/api.py --wallet.name validator --wandb.off --netuid 41 --wallet.hotkey default --subtensor.network test --logging.debug --neuron.run_random_miner_syn_qs_interval 0 --neuron.run_all_miner_syn_qs_interval 0 --neuron.is_disable_tokenizer_reward True --neuron.save_logs True --neuron.only_allowed_miners 5G3hZicxJYGaAeAaVRuebwn2sSrs8twDjtvjXe5cC33G1ySG

.PHONY: run-miner run-api run

run-miner:
	@echo "Running Miner..."
	@$(MINER_CMD)

run-api:
	@echo "Running API..."
	@$(API_CMD)

run:
	@echo "Running both Miner and API..."
	@$(MINER_CMD) &
	@$(API_CMD)
