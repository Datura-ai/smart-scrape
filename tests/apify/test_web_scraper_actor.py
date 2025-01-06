import unittest
from unittest import IsolatedAsyncioTestCase
from neurons.validators.apify.web_scraper_actor import WebScraperActor


class TestWebScraperActor(IsolatedAsyncioTestCase):
    async def test_scrape_metadata(self):
        data = await WebScraperActor().scrape_metadata(
            urls=[
                # "https://www.wirelessworld.com/grips-signal-boosters-and-specialty-products/",  this page does not exist anymore
                "https://www.clinicbarcelona.org/en/news/artistic-creation-to-reduce-anxiety",
                "https://www.nytimes.com/2023/10/31/technology/school-smartphone-bans.html",
                "https://www.reddit.com/r/MobileLegendsGame/comments/1bbxeic/how_good_is_claude_these_days/",
                "https://www.reddit.com/r/OpenAI/comments/183hhbp/is_claude_ai_currently_better_than_chatgpt/",
                "https://www.reddit.com/r/datascience/comments/1e6fpeq/how_much_does_hyperparameter_tuning_actually/",
                "https://arxiv.org/abs/1411.5289v2",
                "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "https://news.ycombinator.com/item?id=26005758",
                "https://news.ycombinator.com/item?id=5674230",
            ]
        )

        expected_data = [
            {
                "title": "Do School Smartphone Bans Work? - The New York Times",
                "description": "Proponents say no-phone rules reduce student distractions and bullying. Critics say the bans could hinder student self-direction and critical thinking.",
                "url": "https://www.nytimes.com/2023/10/31/technology/school-smartphone-bans.html",
            },
            #{
            #    "title": "Grips, Signal Boosters and Specialty Products",
            #    "description": "",
            #    "url": "https://www.wirelessworld.com/grips-signal-boosters-and-specialty-products/",
            #},
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
                "description": "Abstract page for arXiv paper 1411.5289v2: Generalizing the Liveness Based Points-to Analysis",
                "url": "https://arxiv.org/abs/1411.5289v2",
            },
            {
                "title": "Python (programming language) - Wikipedia",
                "description": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation",
                "url": "https://en.wikipedia.org/wiki/Python_(programming_language)"
            },

            {
                "title": "Engineering productivity can be measured, just not how you'd expect | Hacker News",
                "description": """> Just as a sports team wins or loses together, so too should the engineering team be treated as the fundamental unit of success.A sports team has a play book, does your team?  A sports team practices together, does your team?  A sports team works as a unit, does your team?Too many times I have see engineering teams as only a team on the org chart  In reality they solve tickets as individuals with only a small interaction from pull requests.  Otherwise they might as well not even know each other.  They are a team not as in basketball or football, but like golf where once you get to the tee, it's you and only you to get the ball in the hole.""",
                "url": "https://news.ycombinator.com/item?id=26005758"
            },
            {
                "title": "Food Practices Banned in Europe But Allowed in the US | Hacker News",
                "description": """The most sensible European regulations of those listed in the article kindly submitted here are"What Europe did: Banned all forms of animal protein, including chicken litter, in cow feed in 2001."and"What Europe did: In the EU, all antibiotics used in human medicines are banned on farms—and no antibiotics can be used on farms for 'non-medical purposes,' i.e., growth promotion."I'd like to see the United States follow that lead immediately, and I write this as a man who has several uncles and cousins who are farmers, including some who raise cattle. It makes sense to me to have lines of defense against transmission of animal-infecting, and especially antibiotic-resistant-animal-infecting, microbes to human beings, by controlling what animals raised as lifestock eat and how they are treated with veterinary medicines.For the other regulatory practices mentioned in the article, especially washing chicken carcasses, I'd like to see more detailed evidence of the safety trade-offs involved in the practices of the United States and of Europe. I'm less sure on some of the other issues that science actually supports the European practice.""",
                "url": "https://news.ycombinator.com/item?id=5674230"
            }
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
