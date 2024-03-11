import os
from typing import List
import bittensor as bt
from apify_client import ApifyClientAsync
from template.protocol import TwitterScraperTweet

APIFY_API_KEY = os.environ.get("APIFY_API_KEY")


class WebScraperActor:
    def __init__(self) -> None:
        # Actor: https://apify.com/apify/cheerio-scraper
        self.actor_id = "YrQuEkowkNCLdk4j2"
        self.client = ApifyClientAsync(token=APIFY_API_KEY)

    async def scrape_metadata(self, urls: List[str]) -> List[TwitterScraperTweet]:
        if not APIFY_API_KEY:
            bt.logging.warning(
                "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md. This will be required in the next release."
            )
            return []

        try:
            run_input = {
                "debugLog": False,
                "excludes": [{"glob": "/**/*.{png,jpg,jpeg,pdf}"}],
                "forceResponseEncoding": False,
                "ignoreSslErrors": False,
                "keepUrlFragments": False,
                "pageFunction": "async function pageFunction(context) {\n    const { $, request, log } = context;\n\n    // The \"$\" property contains the Cheerio object which is useful\n    // for querying DOM elements and extracting data from them.\n    const pageTitle = $('title').first().text();\n    const pageDescription = $('meta[name=\"description\"]').attr('content');\n\n    // The \"request\" property contains various information about the web page loaded. \n    const url = request.url;\n    \n    // Use \"log\" object to print information to actor log.\n    log.info('Page scraped', { url, pageTitle, pageDescription });\n\n    // Return an object with the data extracted from the page.\n    // It will be stored to the resulting dataset.\n    return {\n        url,\n        pageTitle,\n        pageDescription\n    };\n}",
                "postNavigationHooks": '// We need to return array of (possibly async) functions here.\n// The functions accept a single argument: the "crawlingContext" object.\n[\n    async (crawlingContext) => {\n        // ...\n    },\n]',
                "preNavigationHooks": '// We need to return array of (possibly async) functions here.\n// The functions accept two arguments: the "crawlingContext" object\n// and "requestAsBrowserOptions" which are passed to the `requestAsBrowser()`\n// function the crawler calls to navigate..\n[\n    async (crawlingContext, requestAsBrowserOptions) => {\n        // ...\n    }\n]',
                "proxyConfiguration": {"useApifyProxy": True},
                "startUrls": [{"url": url} for url in urls],
            }

            run = await self.client.actor(self.actor_id).call(run_input=run_input)

            result = []

            async for item in self.client.dataset(
                run["defaultDatasetId"]
            ).iterate_items():
                url = item.get("url", "")
                title = item.get("pageTitle")
                description = item.get("pageDescription")
                result.append({"title": title, "description": description, "url": url})

            return result
        except Exception as e:
            bt.logging.warning(f"Failed to scrape tweets: {e}")
            return []
