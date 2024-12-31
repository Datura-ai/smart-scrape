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
                #"https://arxiv.org/abs/1411.5289v2",
                "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "https://news.ycombinator.com/bestcomments"
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
            #{
            #    "title": "[1411.5289v2] Generalizing the Liveness Based Points-to Analysis",
            #    "description": "The original liveness based flow and context sensitive points-to analysis (LFCPA) is restricted to scalar pointer variables and scalar pointees on stack and static memory. In this paper, we extend it to support heap memory and pointer expressions involving structures, unions, arrays, and pointer arithmetic. The key idea behind these extensions involves constructing bounded names for locations in terms of compile time constants (names and fixed offsets), and introducing sound approximations when it is not possible to do so. We achieve this by defining a grammar for pointer expressions, suitable memory models and location naming conventions, and some key evaluations of pointer expressions that compute the named locations. These extensions preserve the spirit of the original LFCPA which is evidenced by the fact that although the lattices and flow functions change, the overall data flow equations remain unchanged.",
            #    "url": "https://arxiv.org/abs/1411.5289v2",
            #},
            {
                "title": "Python (programming language)",
                "description": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation",
                "url": "https://en.wikipedia.org/wiki/Python_(programming_language)"
            },

            {
                "title": "Best Comments | Hacker News",
                "description": "<3 This has been a work of passion for the past two years of my life (off and on). I hope anyone who uses this can feel the love and care I put into this, and subsequently the amazing private beta community (all ~5,000 strong!) that helped improve and polish this into a better release than I ever could alone.Ghostty got a lot of hype (I cover this in my reflection below), but I want to make sure I call out that there is a good group of EXCELLENT terminals out there, and I'm not claiming Ghostty is strictly better than any of them. Ghostty has different design goals and tradeoffs and if it's right for you great, but if not, you have so many good choices.Shout out to Kitty, WezTerm, Foot in particular. iTerm2 gets some hate for being relatively slow but nothing comes close to touching it in terms of feature count. Rio is a super cool newer terminal, too. The world of terminals is great.I’ve posted a personal reflection here, which has a bit more history on why I started this, what’s next, and some of the takeaways from the past two years. https://mitchellh.com/writing/ghostty-1-0-reflection",
                "url": "https://news.ycombinator.com/bestcomments"
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
