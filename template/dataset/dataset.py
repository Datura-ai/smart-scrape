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
            "What are the emerging trends on Twitter related to {}?",
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
            "#ElonMusk #TechNews",
            "#GenerativeAI",
            "Nvidia",
            "Ukraine",
            "google",
            "Web 3.0",
            "#midjourney",
            "climate change summits",
            "global health initiatives",
            "space exploration missions",
            "electric vehicles",
            "space tourism",
            "renewable energy",
            "virtual reality",
            "stock market trends",
            "cryptocurrency updates",
            "startup ecosystem",
            "e-commerce growth",
            "global trade agreements",
            "election updates",
            "policy changes",
            "political movements",
            "international relations",
            "human rights issues",
            "new movie releases",
            "music album launches",
            "celebrity interviews",
            "streaming service trends",
            "award ceremonies",
            "major sporting events",
            "Olympics updates",
            "championship leagues",
            "athlete profiles",
            "sports technology",
            "fitness trends",
            "mental health awareness",
            "nutritional advice",
            "medical breakthroughs",
            "wellness retreats",
            "travel destinations",
            "fashion weeks",
            "culinary trends",
            "interior design ideas",
            "DIY projects",
            "online education platforms",
            "educational reforms",
            "scholarly achievements",
            "learning methodologies",
            "student initiatives",
            "environmental conservation",
            "scientific discoveries",
            "wildlife preservation",
            "sustainability practices",
            "ecological movements",
            "social justice campaigns",
            "cultural festivals",
            "historical commemorations",
            "social media trends",
            "community outreach programs",
            "art exhibitions",
            "photography contests",
            "music festivals",
            "theatrical performances",
            "literary awards",
            "emerging market trends",
            "global warming impacts",
            "tech startup innovations",
            "urban development projects",
            "adventures in travel blogging",
            "breakthroughs in medicine",
            "evolution of digital currencies",
            "trends in sustainable fashion",
            "advancements in drone technology",
            "discoveries in astrophysics",
            "#SpaceResearch #Astronomy",
            "#RenewableEnergy #GreenTech",
            "#DigitalArt #ArtTech",
            "#FitnessTech #HealthInnovation",
            "#EVs #HybridTech",
            "#ResourceManagementTech #Conservation",
            "#Drones #Innovation"

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