from .db import DBClient
from .dataset_twitter.mock import *

        
tweet_prompts = [
    'Gather opinions on the new iPhone model from tech experts on Twitter.',
    'Find tweets about climate change from the last month.', #+
    'Show me the latest tweets about the SpaceX launch.', #+
    'Collect tweets reacting to the latest UN summit.',
    "Last year's trends  about #openai", #+
    "Tell me last news about elonmusk", #+
    'Tech enthusiasts, share your reviews on the latest iPhone model. How does it compare to previous versions? #iPhoneReview #Technology',
    'Looking for insights from tech experts on the new iPhone model. What are your thoughts on its features and performance? #iPhone #TechReview',
    "Reflecting on the past year, what are the significant developments in climate change we've seen? Share your thoughts. #ClimateChange #YearInReview",
    "Exciting times in space exploration! What are your thoughts on the recent SpaceX launch? #SpaceX #SpaceExploration",
    "The SpaceX launch was a landmark event. How do you think it will impact future space missions? Share your views. #SpaceXLaunch #SpaceNews",
    "What are your key takeaways from the latest UN summit? Discuss the outcomes and their global impact. #UNSummit #GlobalAffairs",
    "Reacting to the recent UN summit: what were the standout moments and decisions? Share your opinions. #UnitedNations #WorldPolitics",
    "Reflecting on the past year, what were the major trends and breakthroughs in #openai? Share your highlights. #AI #TechTrends",
    "Looking back, what were the significant developments in #openai last year that caught your attention? #ArtificialIntelligence #YearInReview",
    "What's the latest buzz around Elon Musk? Share the newest updates and news. #ElonMusk #TechNews",
    "Catch up on the latest happenings with Elon Musk. What’s new and noteworthy? #ElonMuskNews #TechnologyLeaders",
    "What are your thoughts on the latest advancements in renewable energy? Share your insights and opinions. #RenewableEnergy #GreenTech",
    "Calling all gamers! What do you think of the new gaming console releases this year? Share your reviews and experiences. #GamingCommunity #ConsoleReview",
    "As remote work becomes more common, what are the best tools and practices you've discovered? Share your remote work hacks. #RemoteWork #WorkFromHome",
    "With electric cars becoming more popular, what are your experiences with them? Pros, cons, favorite models? Discuss. #ElectricVehicles #EcoFriendly",
    "Exploring the latest in AI: What breakthroughs have impressed you the most recently? Share your thoughts and findings. #ArtificialIntelligence #TechInnovation",
    "Discuss the impact of the latest medical technology advancements on healthcare. How has it changed patient care? #MedTech #HealthcareInnovation",
    "What are the standout fashion trends this season? Share your favorite styles and designers. #FashionTrends #StyleWatch",
    "How has the recent policy changes in education affected learning and teaching? Share your experiences and views. #EducationReform #TeachingAndLearning",
    "What are your predictions for the stock market in the coming months? Share your analysis and insights. #StockMarket #InvestmentTips",
    "Share your favorite travel destinations for 2023. What makes them special? #TravelTips #Wanderlust",
    "Exploring urban development: What are the most innovative and sustainable cities right now? Share your thoughts. #UrbanPlanning #SustainableCities",
    "What are the latest developments in space research and exploration? Share news and opinions. #SpaceResearch #Astronomy",
    "Discuss the impact of social media on modern communication. Has it changed the way we interact? #SocialMedia #DigitalCommunication",
    "What are the newest trends in the world of food and cuisine? Share your favorite recipes and discoveries. #Foodie #CulinaryTrends", #+
    "How are emerging technologies shaping the future of entertainment? Share your thoughts on the latest trends. #TechEntertainment #FutureOfFun"
]