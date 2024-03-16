import random
import datetime
import bittensor as bt
from datasets import load_dataset
from bs4 import BeautifulSoup
import time
import requests


class MockTwitterQuestionsDataset:
    def __init__(self):
        # Extended list of templates for questions
        self.question_templates = [
            "What are the recent {} events?"
            "Tell me the recent news about the {}"
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
            # "What humorous content about {} is being shared on Twitter?",
            "What are Twitter users saying about {}?",
            "What Twitter hashtags are currently associated with {}?",
            "What is the current sentiment about the {}?",
            "How are {} spicing up the recipes in 2024?",
            "What are the recent developments in {}?",
            "Latest advancements in {}",
            "Current trends in {}",
            "Recent discoveries in {}",
            "Updates on {} efforts",
            "New findings in {}",
            "Current events in {}",
            "Latest {} news",
            "Breaking news in {}",
            "Recent achievements in {}",
            "Updates on {} technology",
            "Current best practices in {}",
            "Latest news in {}",
            "New methods in {}",
            "Current state of {}",
            "Latest findings in {}",
            "Updates on {}",
            "Recent innovations in {}",
            "Current trends in {}",
            "What's the latest in {} policy changes?",
            "How is the {} impacting global markets?",
            "What are the newest breakthroughs in {}?",
            "What trends are defining the {} this year?",
            "How are advancements in {} shaping the industry?",
            "What are the main challenges facing the {} today?",
            "How is the {} evolving with technology?",
            "What are the key factors driving {} innovation?",
            "How are {} regulations affecting the market?",
            "What role does {} play in the modern economy?",
            "How is {} influencing consumer behavior?",
            "What are the recent developments in {} startups?",
            "What's new in the world of {}?",
            "How is {} contributing to sustainability?",
            "What are the latest predictions for {} in the coming years?",
            "How is the {} addressing climate change?",
            "What are the emerging technologies in {}?",
            "How is {} affecting international relations?",
            "What are the current trends in {} investment?",
            "What's the future outlook for {}?",
        ]

        # Expanded list of topics, focusing on commonly discussed themes on Twitter

        self.topics = [
            "renewable energy",
            "stock market",
            "artificial intelligence",
            "fashion",
            "space exploration",
            "climate change",
            "nutrition",
            "diet",
            "international politics",
            "movies",
            "entertainment",
            "technology",
            "gadgets",
            "medical research",
            "electric vehicles",
            "software development",
            "education",
            "online learning",
            "sustainable agriculture",
            "economic recovery",
            "psychology",
            "mental health",
            "cybersecurity",
            "data privacy",
            "architecture",
            "design",
            "travel",
            "tourism",
            "USA",
            "tech",
            "innovation",
            "startup",
            "entrepreneurship",
            "world issues",
            "global issues",
            "music",
            "live performances",
            "film",
            "cinema",
            "sport",
            "fitness",
            "fashion trends",
            "gaming",
            "esports",
            "health",
            "wellness",
            "streaming services",
            "cryptocurrency",
            "blockchain",
            "climate sustainability",
            "machine learning",
            "American politics",
            "elections",
            "finance",
            "global politics",
            "diplomacy",
            "Olympics",
            "sports competitions",
            "social media",
            "digital communication",
            "art",
            "culture",
            "healthcare",
            "medical science",
            "technology entrepreneurship",
            "Nvidia AI",
            "Ukraine",
            "geopolitics"
            "Google digital innovation",
            "programming",
            "software development",
            "science",
            "research",
            "history",
            "cultural",
            "cryptocurrency news",
            "blockchain technology",
            "movies",
            "medical innovation",
            "digital health",
            "travel adventure",
            "travel tips",
            "coffee culture",
            "lifestyle",
            "economy",
            "financial markets",
            "internet culture",
            "social media trends",
            "indie games",
            "game design",
            "video game development",
            "technology",
            ".NET framework",
            "programming",
            "Bitcoin",
            "digital currency",
            "AI in fitness",
            "health technology",
            "robotics technology",
            "automation",
            "cinema",
            "movie industry",
            "tech innovation",
            "gadgets",
            "venture capital",
            "geopolitics",
            "artists",
            "directors",
            "competitions",
            "health and wellness",
            "designers",
            "game development",
            "original series",
            "digital currency",
            "green technology",
            "data science",
            "adventure",
            "US news",
            "investing",
            "voting",
            "world events",
            "content creation",
            "exhibitions",
            "SpaceX",
            "international relations",
            "digital services",
            "cloud services",
            "open-source",
            "AI-generated imagery",
            "digital art",
            "cosmos",
            "clean energy",
            "new media art",
            "automotive technology",
            "sustainable practices",
            "unmanned aerial vehicles",
            "digital finance",
            "digital currency trading",
            "relaxation",
            "global economy",
            "viral content",
            "creative gaming",
            "Microsoft",
            "blockchain technology",
            "machine learning applications",
            "Internet of Things",
            "crypto community",
            "fiscal policy",
            "agricultural technology",
            "innovation",
            "virtual reality",
            "affordable housing",
            "mental health awareness",
            "public transportation",
            "e-commerce",
            "renewable energy transition",
            "autonomous vehicles",
            "data privacy",
            "international trade agreements",
            "urban development",
            "quantum computing",
            "global migration patterns",
            "venture capital",
            "space tourism",
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


class StackOverflowDataset:
    def __init__(self):
        # Stack Overflow API endpoint for a random article
        self.url = "https://api.stackexchange.com/2.3/questions"
        self.questions = []

    def get_stack_questions(self):
        url = "https://api.stackexchange.com/2.3/questions"
        params = {
            "order": "desc",
            "sort": "votes",  # Sorting by votes means that it's likely that the same questions will be fetched again
            "site": "stackoverflow",
            "pagesize": 100,  # Fetch 100 questions per API call
            "page": random.randint(1, 5),
        }

        # Fetch questions
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Parse response
        questions = response.json()["items"]

        # Filter questions by minimum upvotes
        min_upvotes = 10
        filtered_questions = [q for q in questions if q["score"] >= min_upvotes]
        # Shuffle the questions
        random.shuffle(filtered_questions)

        # Add the questions to the list of questions
        self.questions.extend(filtered_questions)
        return

    def get_stack_question(self) -> dict:
        # If the list of questions is empty, fetch more questions
        if not self.questions:
            self.get_stack_questions()
        question = self.questions.pop()
        # Fetch the highest voted answer for the selected question
        answer = self.get_stack_answer(question)
        return {"question": question["title"], "answer": answer}

    def get_stack_answer(self, question):
        question_id = question["question_id"]
        url_answers = (
            f"https://api.stackexchange.com/2.3/questions/{question_id}/answers"
        )
        params_answers = {
            "order": "desc",
            "sort": "votes",
            "site": "stackoverflow",
            "filter": "withbody",  #'!9_bDDxJY5'
        }
        response_answers = requests.get(url_answers, params=params_answers)
        response_answers.raise_for_status()
        answers = response_answers.json()["items"]
        if not answers:
            bt.logging.warning("No answers found for the question!")

        highest_voted_answer = answers[0]  # The first answer is the highest voted
        soup = BeautifulSoup(highest_voted_answer["body"], "html.parser")
        full_content = soup.get_text(separator="\n")
        return full_content

    def next(self):
        bt.logging.debug("Retrieving data from prompting.dataset...")
        t0 = time.time()
        info = self.get_stack_question()
        info["fetch_time"] = time.time() - t0
        return info


if __name__ == "__main__":
    # Example usage
    twitter_questions_dataset = MockTwitterQuestionsDataset()
    for _ in range(100):
        print(twitter_questions_dataset.next())
