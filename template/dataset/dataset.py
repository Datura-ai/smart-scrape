import random

class MockTwitterQuestionsDataset:
    def __init__(self):
        # Extended list of templates for questions
        self.question_templates = [
            "What's the current sentiment on Twitter about {}?",
            "Are there any breaking news tweets about {}?",
            "What are the latest viral tweets about {}?",
            "What's trending on Twitter about {}?",
            "What's the latest discussion on Twitter about {}?",
            "Are there any new developments in {} on Twitter?",
            "What are the recent Twitter polls about {}?",
            "How are Twitter users reacting to {}?",
            "What are the recent tweets regarding {}?",
            "What are the top opinions on Twitter about {}?",
            "How is the Twitter community responding to {}?",
            "What humorous content about {} is being shared on Twitter?",
            "What are Twitter influencers saying about {}?",
            "What Twitter hashtags are currently associated with {}?",
            "How is {} being represented in Twitter memes?",
            "How is {} being represented in Twitter memes?",
            "What are the emerging trends on Twitter related to {}?",
            "How are {} related topics evolving on Twitter today?",
            "What Twitter threads are unfolding about {}?",
        ]

        # Expanded list of topics, focusing on commonly discussed themes on Twitter
        self.topics = [
            "tech",
            "startup",
            "world",
            "music",
            "film",
            "sport",
            "fitness trends",
            "fashion highlights",
            "gaming",
            "health",
            "fashion",
            "Netflix",
            "cryptocurrency",
            "environmental",
            "robotics",
            "sports",
            "gaming",
            "climate",
            "ai",
            "tourism",
            "USA",
            "stock market",
            "election",
            "policy",
            "political",
            "international",
            "movie",
            "music",
            "service",
            "award",
            "Olympics",
            "health",
            "social",
            "cultural",
            "social media",
            "community",
            "art",
            "festivals",
            "medicine",
            "currencies",
            "astrophysics",
            "#ai",
            "#openai",
            "#dev",
            "#crypto",
            "#blockchain",
            "#technology",
            "#healthcare",
            "#education",
            "#entertainment",
            "#travel",
            "#foodie",
            "#sports",
            "#music",
            "#movies",
            "#gaming",
            "#literature",
            "#science",
            "#history",
            "#art",
            "#culture",
            "#goverment",
            "#python",
            "#js",
            "#btc",
            "#crypto",
            "#dao",
            "#eth",
            "#llm",
            "#eth",
            "#america",
            "#eth",
            "#NY",
            "#code",
            "#chatgpt",
            "#aiart",
            "Elon Musk",
            "#Biotech #MedicalScience",
            "#VR #TechForecast",
            "#IoT",
            "#AIFitness #HealthTech",
            "#TravelHacks #Adventure",
            "#ElonMusk #TechNews"
        ]

    def generate_question(self):
        # Randomly select a question template and a topic
        template = random.choice(self.question_templates)
        topic = random.choice(self.topics)

        # Generate a question
        return template.format(topic)

    def next(self):
        # Return a generated question
        return self.generate_question()


if __name__ == "__main__":
    # Example usage
    twitter_questions_dataset = MockTwitterQuestionsDataset()
    for _ in range(100):
        print(twitter_questions_dataset.next())
