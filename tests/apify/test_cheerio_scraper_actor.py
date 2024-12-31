import unittest
from unittest import IsolatedAsyncioTestCase
from neurons.validators.apify.cheerio_scraper_actor import CheerioScraperActor


class TestCheerioScraperActor(IsolatedAsyncioTestCase):
    async def test_scrape_metadata(self):
        data = await CheerioScraperActor().scrape_metadata(
            urls=[
                "https://www.nytimes.com/2023/10/31/technology/school-smartphone-bans.html",
                "https://apnews.com/article/usa-basketball-serbia-paris-olympics-c83403938291464a13d83d54210eeb0c",
                "https://hokiesports.com/news/2024/07/18/mens-basketball-single-game-tickets-now-available",
                #"https://arxiv.org/abs/1411.5289v2",
                "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "https://news.ycombinator.com/bestcomments"
            ]
        )

        expected_data = [
            {
                "title": "Stephen Curry strong in US men's basketball team's 105-79 win over Serbia | AP News",
                "description": "Stephen Curry scored 24 points, Bam Adebayo added 17 and the U.S. beat Serbia 105-79 to improve to 3-0 in its five-game slate of exhibitions before the Paris Olympics.",
                "url": "https://apnews.com/article/usa-basketball-serbia-paris-olympics-c83403938291464a13d83d54210eeb0c",
            },
            {
                "title": "Do School Smartphone Bans Work? - The New York Times",
                "description": "Proponents say no-phone rules reduce student distractions and bullying. Critics say the bans could hinder student self-direction and critical thinking.",
                "url": "https://www.nytimes.com/2023/10/31/technology/school-smartphone-bans.html",
            },
            {
                "title": "Men’s basketball single-game tickets now available - Virginia Tech Athletics",
                "description": "",
                "url": "https://hokiesports.com/news/2024/07/18/mens-basketball-single-game-tickets-now-available",
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
