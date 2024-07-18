import unittest
from unittest import IsolatedAsyncioTestCase
from neurons.validators.apify.web_scraper_actor import WebScraperActor


class TestWebScraperActor(IsolatedAsyncioTestCase):
    async def test_scrape_metadata(self):
        data = await WebScraperActor().scrape_metadata(
            urls=[
                "https://www.wirelessworld.com/grips-signal-boosters-and-specialty-products/",
                "https://www.clinicbarcelona.org/en/news/artistic-creation-to-reduce-anxiety",
                "https://www.nytimes.com/2023/10/31/technology/school-smartphone-bans.html",
                "https://www.reddit.com/r/MobileLegendsGame/comments/1bbxeic/how_good_is_claude_these_days/",
                "https://www.reddit.com/r/OpenAI/comments/183hhbp/is_claude_ai_currently_better_than_chatgpt/",
                "https://www.reddit.com/r/singularity/comments/1b922bo/claude_is_really_impressive/",
            ]
        )

        expected_data = [
            {
                "title": "Do School Smartphone Bans Work? - The New York Times",
                "description": "Proponents say no-phone rules reduce student distractions and bullying. Critics say the bans could hinder student self-direction and critical thinking.",
                "url": "https://www.nytimes.com/2023/10/31/technology/school-smartphone-bans.html",
            },
            {
                "title": "Grips, Signal Boosters and Specialty Products",
                "description": None,
                "url": "https://www.wirelessworld.com/grips-signal-boosters-and-specialty-products/",
            },
            {
                "title": "Art Therapy to Reduce Anxiety | PortalCLÍNIC",
                "description": "Painting, drawing or sculpture are part of the treatment of people with mental health problems such as anxiety. This method of occupational therapy provide",
                "url": "https://www.clinicbarcelona.org/en/news/artistic-creation-to-reduce-anxiety",
            },
            {
                "title": "How good is claude these days? : r/MobileLegendsGame",
                "description": "I follow the rule of eps. I main heroee with the epic skins i have.. unless theyre an assassin or bad",
                "url": "https://www.reddit.com/r/MobileLegendsGame/comments/1bbxeic/how_good_is_claude_these_days/",
            },
            {
                "title": "Is Claude AI currently better than chatGPT? : r/OpenAI",
                "description": "I was doing some research and came across Claud AI, can anyone who has already used both Claud and ChatGPT tell me if it is better and how it differs from chatGPT?",
                "url": "https://www.reddit.com/r/OpenAI/comments/183hhbp/is_claude_ai_currently_better_than_chatgpt/",
            },
            {
                "title": "Claude is really impressive : r/singularity",
                "description": "I feel bad just leaving him hanging like that, but I need a lot more time to write up a response than he does",
                "url": "https://www.reddit.com/r/singularity/comments/1b922bo/claude_is_really_impressive/",
            },
        ]

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), len(expected_data))

        for item in data:
            self.assertIsInstance(item, dict)

            expected_item = next(
                expected for expected in expected_data if expected["url"] == item["url"]
            )

            self.assertIsInstance(expected_item, dict)
            self.assertEqual(expected_item["title"], item["title"])
            self.assertEqual(expected_item["url"], item["url"])
            self.assertEqual(expected_item["description"], item["description"])


if __name__ == "__main__":
    unittest.main()
