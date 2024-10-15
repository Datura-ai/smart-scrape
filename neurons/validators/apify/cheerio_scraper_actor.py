import os
from typing import List
import bittensor as bt
from apify_client import ApifyClientAsync
from datura.protocol import TwitterScraperTweet

APIFY_API_KEY = os.environ.get("APIFY_API_KEY")


class CheerioScraperActor:
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

        if not urls:
            return []

        try:
            run_input = {
                "debugLog": False,
                "excludes": [{"glob": "/**/*.{png,jpg,jpeg}"}],
                "forceResponseEncoding": False,
                "ignoreSslErrors": False,
                "keepUrlFragments": False,
                "pageFunction": "async function pageFunction(context) {\n    const { $, request, log } = context;\n\n    // Extract the page title\n    const pageTitle = $('title').first().text();\n\n    // Extract the meta description\n    let description = $('meta[name=\"description\"]').attr('content') || '';\n\n    if (!description) {\n        description = $('meta[property=\"og:description\"]').attr('content')\n    }\n\n    // Get the URL from the request object\n    const url = request.url;\n\n    // Return an object with the extracted data\n    return {\n        url,\n        pageTitle,\n        description,\n    };\n}",
                "postNavigationHooks": '// We need to return array of (possibly async) functions here.\n// The functions accept a single argument: the "crawlingContext" object.\n[\n    async (crawlingContext) => {\n        // ...\n    },\n]',
                "preNavigationHooks": '// We need to return array of (possibly async) functions here.\n// The functions accept two arguments: the "crawlingContext" object\n// and "requestAsBrowserOptions" which are passed to the `requestAsBrowser()`\n// function the crawler calls to navigate..\n[\n    async (crawlingContext, requestAsBrowserOptions) => {\n        // ...\n    }\n]',
                "proxyConfiguration": self.get_proxy_configuration(urls),
                "proxyRotation": "RECOMMENDED",
                "startUrls": [{"url": url} for url in urls],
            }

            run = await self.client.actor(self.actor_id).call(
                run_input=run_input,
                memory_mbytes=512,
            )

            result = []

            async for item in self.client.dataset(
                run["defaultDatasetId"]
            ).iterate_items():
                url = item.get("url", "")
                title = item.get("pageTitle")
                description = item.get("description")
                result.append({"title": title, "description": description, "url": url})

            return result
        except Exception as e:
            bt.logging.warning(
                f"CheerioScraperActor: Failed to scrape web links {urls}: {e}"
            )
            return []

    def get_proxy_configuration(self, urls: List[str]) -> dict:
        """
        Returns the proxy configuration based on the URLs to scrape.
        Used to save scraping costs as some urls don't need residential proxies.
        """

        residential = True

        if all(
            any(
                domain in url
                for domain in ["news.ycombinator.com", "wikipedia.org", "arxiv.org"]
            )
            for url in urls
        ):
            residential = False

        proxy_configuration = {
            "useApifyProxy": True,
            "apifyProxyGroups": [],
        }

        if residential:
            proxy_configuration["apifyProxyGroups"].append("RESIDENTIAL")

        return proxy_configuration
