#!/bin/bash

# Initialize variables
script="datura/scripts/bittensor_docs_indexer.py"
proc_name="bittensor_documentation_indexing_process"
python_script="python3 $script"
autoRunLoc=$(readlink -f "$0")
args=()

old_args=$@

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
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

if pm2 status | grep -q $proc_name; then
    echo "The script is already running with pm2. Stopping and restarting..."
    pm2 delete $proc_name
fi

joined_args="${args[*]}"

echo "Running $script with the following arguments: $joined_args"

pm2 start "$python_script" --name "$proc_name" -- $joined_args

echo "Process started with pm2"

