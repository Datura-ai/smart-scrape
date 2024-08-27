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
                "https://www.reddit.com/r/datascience/comments/1e6fpeq/how_much_does_hyperparameter_tuning_actually/",
                "https://arxiv.org/abs/1411.5289v2",
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
                "description": "",
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
                "title": "How much does hyperparameter tuning actually matter : r/datascience",
                "description": 'I say this as in: yes obvioisly if you set ridiculous values for your learning rate and batch sizes and penalties or whatever else, obviously your model will be ass.But once you arrive at a set of "reasonable" hyper parameters, as in theyre probably not globally optimal or even close but they produce OK results and is pretty close to what you normally see in papers. How much gain is there to be had from tuning hyper parameters extensively?',
                "url": "https://www.reddit.com/r/datascience/comments/1e6fpeq/how_much_does_hyperparameter_tuning_actually/",
            },
            {
                "title": "[1411.5289v2] Generalizing the Liveness Based Points-to Analysis",
                "description": "The original liveness based flow and context sensitive points-to analysis (LFCPA) is restricted to scalar pointer variables and scalar pointees on stack and static memory. In this paper, we extend it to support heap memory and pointer expressions involving structures, unions, arrays, and pointer arithmetic. The key idea behind these extensions involves constructing bounded names for locations in terms of compile time constants (names and fixed offsets), and introducing sound approximations when it is not possible to do so. We achieve this by defining a grammar for pointer expressions, suitable memory models and location naming conventions, and some key evaluations of pointer expressions that compute the named locations. These extensions preserve the spirit of the original LFCPA which is evidenced by the fact that although the lattices and flow functions change, the overall data flow equations remain unchanged.",
                "url": "https://arxiv.org/abs/1411.5289v2",
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
