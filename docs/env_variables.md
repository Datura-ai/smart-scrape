# Enhanced Guide for Setting Up Environment Variables in Smart-Scrape System

## Detailed Steps for Environment Variable Configuration

### Prerequisites
- Access to a terminal interface.
- Accounts on OpenAI, Weights & Biases, and Twitter Developer Portal.

### Setting Up Variables
1. **OPENAI_API_KEY**
   - **Usage**: Authenticates with the OpenAI API.
   - **How to obtain**: Visit [OpenAI API](https://beta.openai.com/signup/), sign up or log in, navigate to the API section, and generate a key.

2. **WANDB_API_KEY**
   - **Usage**: For experiment tracking with Weights & Biases.
   - **How to obtain**: Sign up or log in at [Weights & Biases](https://wandb.ai/), and generate a key in the API keys section of your account settings.

3. **TWITTER_BEARER_TOKEN**
   - **Usage**: Accesses the Twitter API.
   - **How to obtain**: Create a Twitter Developer account, create an app at [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard), and generate a token in the "Keys and Tokens" section.

4. **VALIDATOR_ACCESS_KEY**
   - **Usage**: Allows service access to the validator.
   - **Note**: This can be any unique, strong, and random string for security.

### Executing Commands
Open a terminal and run the following commands. Replace the placeholders with your actual keys.

```bash
export OPENAI_API_KEY="<your_openai_api_key>"
export WANDB_API_KEY="<your_wandb_api_key>"
export TWITTER_BEARER_TOKEN="<your_twitter_bearer_token>"
export VALIDATOR_ACCESS_KEY="<your_validator_access_key>"
