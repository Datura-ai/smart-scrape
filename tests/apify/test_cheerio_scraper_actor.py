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
