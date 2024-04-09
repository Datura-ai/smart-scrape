import unittest
from neurons.validators.penalty.link_validation import LinkValidationPenaltyModel
from neurons.validators.utils.tasks import TwitterTask
from datura.protocol import ScraperStreamingSynapse

completion1 = f"""
Last year's recipe trends from verified accounts highlighted veganism, innovative food products, and sustainable agriculture. Key insights are supported by these specific Twitter links:
- Veganuary 2024 trend: [Tweet by @XtalksFood](https://twitter.com/XtalksFood/status/1743286252969828589)
- New food products in 2024: [Tweet by @XtalksFood](https://twitter.com/XtalksFood/status/1742562108363952545)
These links directly corroborate the identified trends and provide concrete examples from Twitter.
"""

completion2 = f"""
In 2023, one of the most influential political commentators on Twitter was John Doe (@JohnDoe). He was known for his insightful tweets on various political events. See:
- [Tweet by @JohnDoe](https://twitter.com/nononno/status/1122334455) on recent election analysis.
- [Tweet by @PoliticalDigest](https://twitter.com/nononno/status/1122334455) mentioning John Doe's influence.
However, the answer lacks broader context about his influence compared to other commentators.
"""

completion3 = f"""
In 2023, one of the most influential political commentators on Twitter was John Doe (@JohnDoe). He was known for his insightful tweets on various political events. See:
- Veganuary 2024 trend: [Tweet by @XtalksFood](https://twitter.com/XtalksFood/status/1743286252969828589)
- New food products in 2024: [Tweet by @XtalksFood](https://twitter.com/XtalksFood/status/1742562108363952545)
- [Tweet by @PoliticalDigest](https://twitter.com/nononno/status/1122334455) mentioning John Doe's influence.
- [Tweet by @PoliticalDigest](https://twitter.com/nononno/status/1122334455) mentioning John Doe's influence.
However, the answer lacks broader context about his influence compared to other commentators.
"""


class LinkValidationPenaltyModelTestCase(unittest.TestCase):
    """
    This class contains unit tests for the LinkValidationPenaltyModel class.
    """

    def setUp(self):
        self.max_penalty = 1.0
        self.validator = LinkValidationPenaltyModel(self.max_penalty)

    def test_find_twitter_links(self):
        """
        Test if the find_twitter_links method correctly identifies Twitter links in a text.
        """
        text_with_links = "Check out this tweet https://twitter.com/user/status/123 and this one https://x.com/user/status/456"
        expected_links = [
            "https://twitter.com/user/status/123",
            "https://x.com/user/status/456",
        ]
        found_links = self.validator.find_twitter_links(text_with_links)
        self.assertEqual(found_links, expected_links)

    def test_is_valid_twitter_link(self):
        """
        Test if the is_valid_twitter_link method correctly validates Twitter links.
        """
        valid_link = "https://twitter.com/user/status/123"
        invalid_link = "https://nottwitter.com/user/status/123"
        self.assertTrue(self.validator.is_valid_twitter_link(valid_link))
        self.assertFalse(self.validator.is_valid_twitter_link(invalid_link))

    def test_extract_tweet_id(self):
        """
        Test if the extract_tweet_id method correctly extracts the tweet ID from a URL.
        """
        url = "https://twitter.com/user/status/123"
        expected_tweet_id = "123"
        tweet_id = self.validator.extract_tweet_id(url)
        self.assertEqual(tweet_id, expected_tweet_id)

    def test_calculate_penalties(self):
        """
        Test if the calculate_penalties method returns correct penalties for given responses.
        """
        task = TwitterTask(
            base_text="Some base text for relevance",
            task_name="test",
            task_type="test",
            criteria=[],
        )
        responses = [
            ScraperStreamingSynapse(
                completion=completion1,
                messages="",
                model="",
                seed=1,
                completion_links=[
                    "https://twitter.com/XtalksFood/status/1743286252969828589",
                    "https://twitter.com/XtalksFood/status/1743286252969828589",
                ],
            ),
            ScraperStreamingSynapse(
                completion=completion2, messages="", model="", seed=1
            ),
            ScraperStreamingSynapse(
                completion="This is a tweet with no link", messages="", model="", seed=1
            ),
            ScraperStreamingSynapse(
                completion=completion3, messages="", model="", seed=1
            ),
        ]
        expected_penalties = [self.max_penalty, 0.0, 0.0, 0.5]
        penalties = self.validator.calculate_penalties(task, responses)
        for penalty, expected_penalty in zip(penalties, expected_penalties):
            self.assertEqual(penalty.item(), expected_penalty)


if __name__ == "__main__":
    unittest.main()
