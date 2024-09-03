import os
from typing import List
import bittensor as bt
from apify_client import ApifyClientAsync
from datura.protocol import TwitterScraperTweet

APIFY_API_KEY = os.environ.get("APIFY_API_KEY")


class WebScraperActor:
    def __init__(self) -> None:
        # Actor: https://apify.com/apify/web-scraper
        self.actor_id = "moJRLRc85AitArpNN"
        self.client = ApifyClientAsync(token=APIFY_API_KEY)

    async def scrape_metadata(self, urls: List[str]) -> List[TwitterScraperTweet]:
        if not APIFY_API_KEY:
            bt.logging.warning(
                "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md. This will be required in the next release."
            )
            return []

        if not urls:
            return []

        try:
            # Web scraper
            run_input = {
                "debugLog": False,
                "breakpointLocation": "NONE",
                "browserLog": False,
                "closeCookieModals": False,
                "downloadCss": False,
                "downloadMedia": False,
                "headless": True,
                "ignoreCorsAndCsp": False,
                "ignoreSslErrors": False,
                "injectJQuery": True,
                "keepUrlFragments": False,
                "pageFunction": "async function pageFunction(context) {\n  const $ = context.jQuery;\n  const pageTitle = $('title').first().text();\n  let description = $('meta[name=\"description\"]').attr('content') || '';\n\n  if (!description) {\n      description = $('meta[property=\"og:description\"]').attr('content') || '';\n  }\n\n  const isRedditUrl = context.request.url.includes('reddit.com');\n  \n  if (isRedditUrl) {\n    // Extract content from the first p in the first div.text-neutral-content\n    description = $('div.text-neutral-content').first().find('p').map(function() {\n      return $(this).text().trim();\n    }).get().join('');\n  }\n\n  return {\n    url: context.request.url,\n    pageTitle,\n    description,\n  };\n}",
                "postNavigationHooks": '// We need to return array of (possibly async) functions here.\n// The functions accept a single argument: the "crawlingContext" object.\n[\n    async (crawlingContext) => {\n        // ...\n    },\n]',
                "preNavigationHooks": '// We need to return array of (possibly async) functions here.\n// The functions accept two arguments: the "crawlingContext" object\n// and "gotoOptions".\n[\n    async (crawlingContext, gotoOptions) => {\n        // ...\n    },\n]\n',
                "proxyConfiguration": {
                    "useApifyProxy": True,
                    "apifyProxyGroups": ["RESIDENTIAL"],
                },
                "runMode": "PRODUCTION",
                "startUrls": [{"url": url} for url in urls],
                "useChrome": False,
                "waitUntil": ["networkidle2"],
            }

            run = await self.client.actor(self.actor_id).call(run_input=run_input)

            result = []

            async for item in self.client.dataset(
                run["defaultDatasetId"]
            ).iterate_items():
                url = item.get("url", "")
                title = item.get("pageTitle")
                description = item.get("description")
                result.append(
                    {
                        "title": title,
                        "description": description,
                        "url": url,
                    }
                )

            return result
        except Exception as e:
            bt.logging.warning(
                f"WebScraperActor: Failed to scrape web links {urls}: {e}"
            )
            return []
