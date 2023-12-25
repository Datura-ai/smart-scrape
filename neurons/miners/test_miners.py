import subprocess

num_miners = 5
base_port = 10000
start_num = 1
wallet_name = 1
pm2_commands = []

for i in range(num_miners):
    miner_number = start_num + i
    port = base_port + i

    # Construct the PM2 start command
    # command = f"pm2 start miner.py --interpreter python3 --name {wallet_name}:{miner_number} -- --wallet.name {wallet_name} --wallet.hotkey {miner_number} --subtensor.network finney --netuid 18 --axon.port {port*wallet_name} --logging.debug"
    # Python start command
    command = f"python3 miner.py --wallet.name {wallet_name} --wallet.hotkey {miner_number} --subtensor.network test --netuid 24 --axon.port {port*wallet_name-1000} --logging.debug"

    pm2_commands.append(command)

for cmd in pm2_commands:
    print(cmd)
# for cmd in pm2_commands:
#     print(f"Executing: {cmd}")
#     subprocess.run(cmd, shell=True, check=True)
