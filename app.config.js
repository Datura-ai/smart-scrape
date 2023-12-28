module.exports = {
  apps : [{
    name   : 'smart_scrape_validators_main_process',
    script : 'neurons/validators/validator.py',
    interpreter: 'python3',
    min_uptime: '5m',
    max_restarts: '5',
    args: ['--wallet.name','validator','--wallet.hotkey','default','--subtensor.network','test']
  }]
}
