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
                "https://news.ycombinator.com/item?id=26005758",
                "https://news.ycombinator.com/item?id=5674230",
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
