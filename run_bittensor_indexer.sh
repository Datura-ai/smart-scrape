#!/bin/bash

# Initialize variables
script_path="datura/scripts/bittensor_docs_indexer.py"
script_dir="$(dirname "$0")"
script="$(realpath "$script_dir/$script_path")"
proc_name="bittensor_documentation_indexing_process"
python_script="python3 $script"
autoRunLoc=$(realpath "$0")
args=()
old_args=$@

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "Error: pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi

# Check if the script file exists
if [[ ! -f "$script" ]]; then
    echo "Error: The script file '$script' does not exist."
    exit 1
fi

# Loop through all command line arguments
while [[ $# -gt 0 ]]; do
    arg="$1"

    # Check if the argument is a flag
    if [[ "$arg" == --* ]]; then
        # Add the flag to the args array
        args+=("$arg")
        shift
    else
        # Argument is not a flag, add it to the args array
        args+=("'$arg'")
        shift
    fi
done

echo "pm2 process name: $proc_name"

# Clean up any leftover processes
pm2 stop all
pm2 delete all

# Join the arguments with commas using printf
joined_args=$(printf "%s," "${args[@]}")

# Remove the trailing comma (if any)
joined_args=${joined_args%,}

# If no arguments were provided, set joined_args to an empty array
if [ -z "$joined_args" ]; then
    joined_args="[]"
fi

# Create a temporary configuration file
config_file=$(mktemp)
echo "module.exports = {
  apps : [{
    name   : '$proc_name',
    script : '$python_script',
    min_uptime: '5m',
    max_restarts: '5',
    args: $joined_args
  }]
}" > "$config_file"

# Print the configuration to be used
cat "$config_file"

# Start the process with pm2 using the temporary configuration file
echo "Running $script with the following pm2 config:"
pm2 start "$config_file" --no-daemon

# Clean up the temporary configuration file
rm "$config_file"

echo "Process started with pm2"
