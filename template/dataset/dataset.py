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
            "What are Twitter users saying about {}?",
            "What Twitter hashtags are currently associated with {}?",
            # "How is {} being represented in Twitter memes?",
            # "What are the emerging trends on Twitter related to {}?",
            
        ]

        # Expanded list of topics, focusing on commonly discussed themes on Twitter

        self.topics = [
            "tech, innovation, gadgets, software, hardware, #TechTrends, #Innovation, #Gadgets, #Software, #Hardware",
            "startup, entrepreneurship, business model, venture capital, startup culture, #StartupLife, #Entrepreneurship, #BusinessModel, #VentureCapital, #StartupCulture",
            "world, global issues, international news, geopolitics, cultural diversity, #GlobalNews, #InternationalAffairs, #Geopolitics, #CulturalDiversity, #WorldNews",
            "music, genres, live performances, music industry, artists, #MusicLife, #LiveMusic, #MusicIndustry, #Artists, #Genres",
            "film, movies, cinema, film industry, directors, #Film, #Movies, #Cinema, #FilmIndustry, #Directors",
            "sport, athletics, competitions, sports news, fitness, #Sport, #Athletics, #Competitions, #SportsNews, #Fitness",
            "fitness trends, health and wellness, workout routines, gym culture, fitness technology, #FitnessTrends, #HealthWellness, #WorkoutRoutines, #GymCulture, #FitnessTech",
            "fashion highlights, trends, designers, fashion shows, style, #FashionHighlights, #Trends, #Designers, #FashionShows, #Style",
            "gaming, esports, video games, game development, gaming culture, #Gaming, #Esports, #VideoGames, #GameDevelopment, #GamingCulture",
            "health, wellness, medicine, healthcare trends, fitness, #Health, #Wellness, #Medicine, #HealthcareTrends, #Fitness",
            "Netflix, streaming services, original series, movies, entertainment, #Netflix, #StreamingServices, #OriginalSeries, #Movies, #Entertainment",
            "cryptocurrency, blockchain, digital currency, crypto trading, crypto mining, #Cryptocurrency, #Blockchain, #DigitalCurrency, #CryptoTrading, #CryptoMining",
            "climate, environmental issues, sustainability, climate change, green technology, #Climate, #EnvironmentalIssues, #Sustainability, #ClimateChange, #GreenTech",
            "ai, artificial intelligence, machine learning, AI applications, data science, #AI, #ArtificialIntelligence, #MachineLearning, #AIApplications, #DataScience",
            "tourism, travel, adventure, destinations, travel hacks, #Tourism, #Travel, #Adventure, #Destinations, #TravelHacks",
            "USA, American politics, US news, elections, policy, #USA, #AmericanPolitics, #USNews, #Elections, #Policy",
            "stock market, finance, investing, financial news, markets, #StockMarket, #Finance, #Investing, #FinancialNews, #Markets",
            "election, politics, political campaigns, voting, government, #Election, #Politics, #PoliticalCampaigns, #Voting, #Government",
            "international, global politics, world events, diplomacy, international relations, #International, #GlobalPolitics, #WorldEvents, #Diplomacy, #InternationalRelations",
            "Olympics, sports competitions, athletes, international sports, Olympic Games, #Olympics, #SportsCompetitions, #Athletes, #InternationalSports, #OlympicGames",
            "social media, digital communication, online communities, social networking, content creation, #SocialMedia, #DigitalCommunication, #OnlineCommunities, #SocialNetworking, #ContentCreation",
            "art, culture, exhibitions, artists, galleries, #Art, #Culture, #Exhibitions, #Artists, #Galleries",
            "medicine, healthcare, medical science, health innovation, medical research, #Medicine, #Healthcare, #MedicalScience, #HealthInnovation, #MedicalResearch",
            "Elon Musk, technology innovation, SpaceX, Tesla, entrepreneurship, #ElonMusk, #TechnologyInnovation, #SpaceX, #Tesla, #Entrepreneurship",
            "Nvidia, technology, graphics processing units, AI, computing, #Nvidia, #Technology, #GPUs, #AI, #Computing",
            "Ukraine, international news, geopolitics, conflict, global affairs, #Ukraine, #InternationalNews, #Geopolitics, #Conflict, #GlobalAffairs",
            "google, technology, search engines, digital innovation, online services, #Google, #Technology, #SearchEngines, #DigitalInnovation, #OnlineServices",
            "programming, coding, software development, technology, #Programming, #Coding, #SoftwareDevelopment, #Technology, #Dev",
            "science, research, innovation, scientific discoveries, technology, #Science, #Research, #Innovation, #ScientificDiscoveries, #Technology",
            "history, cultural heritage, historical events, education, learning, #History, #CulturalHeritage, #HistoricalEvents, #Education, #Learning",
            "cryptos, cryptocurrency news, blockchain technology, digital finance, crypto community, #Cryptos, #CryptocurrencyNews, #BlockchainTechnology, #DigitalFinance, #CryptoCommunity",
            "entertainment, movies, TV shows, streaming content, celebrity news, #Entertainment, #Movies, #TVShows, #StreamingContent, #CelebrityNews",
            "healthtech, medical innovation, healthcare technology, fitness apps, digital health, #HealthTech, #MedicalInnovation, #HealthcareTechnology, #FitnessApps, #DigitalHealth",
            "travel and adventure, destinations, travel tips, adventure tourism, travel photography, #TravelAndAdventure, #Destinations, #TravelTips, #AdventureTourism, #TravelPhotography",
            "#government, politics, policy, regulation, public administration, #Politics, #Policy, #Regulation, #PublicAdministration",
            "#python, programming language, software development, data science, machine learning, #ProgrammingLanguage, #SoftwareDevelopment, #DataScience, #MachineLearning",
            "#js, JavaScript, web development, front-end, programming, #JavaScript, #WebDevelopment, #FrontEnd, #Programming",
            "#btc, Bitcoin, cryptocurrency, blockchain, digital currency, #Bitcoin, #Cryptocurrency, #Blockchain, #DigitalCurrency",
            "#crypto, cryptocurrency, blockchain, digital assets, decentralized finance, #Cryptocurrency, #Blockchain, #DigitalAssets, #DeFi",
            "#dao, decentralized autonomous organization, blockchain, smart contracts, governance, #Decentralized, #Blockchain, #SmartContracts, #Governance",
            "#eth, Ethereum, smart contracts, blockchain, decentralized apps, #Ethereum, #SmartContracts, #Blockchain, #DApps",
            "#llm, large language models, AI, machine learning, natural language processing, #LargeLanguageModels, #AI, #MachineLearning, #NLP",
            "#america, USA, politics, culture, society, #USA, #Politics, #Culture, #Society",
            "#JustifyMyLove #JML #NY #Live #HD #90s #Music, Madonna, music video, New York, live performance, #Madonna, #MusicVideo, #NewYork, #LivePerformance",
            "#code, programming, software development, coding practices, technology, #Programming, #SoftwareDevelopment, #Coding, #Technology",
            "#chatgpt, conversational AI, natural language processing, OpenAI, chatbots, #ConversationalAI, #NLP, #OpenAI, #Chatbots",
            "#aiart, artificial intelligence, digital art, generative art, creativity, #ArtificialIntelligence, #DigitalArt, #GenerativeArt, #Creativity",
            "Elon Musk, technology entrepreneur, SpaceX, Tesla, innovation, #ElonMusk, #SpaceX, #Tesla, #Innovation, #TechnologyEntrepreneur",
            "#Biotech #MedicalScience, biotechnology, medical research, healthcare innovation, science, #Biotechnology, #MedicalResearch, #HealthcareInnovation, #Science",
            "#VR #TechForecast, virtual reality, technology trends, immersive technology, future tech, #VirtualReality, #TechnologyTrends, #ImmersiveTechnology, #FutureTech",
            "#IoT, Internet of Things, smart devices, connectivity, technology, #InternetOfThings, #SmartDevices, #Connectivity, #Technology",
            "#AIFitness #HealthTech, artificial intelligence, fitness technology, health innovation, personal wellness, #ArtificialIntelligence, #FitnessTechnology, #HealthInnovation, #PersonalWellness",
            "#TravelHacks #Adventure #trip, travel tips, adventure travel, travel planning, destination guides, #TravelTips, #AdventureTravel, #TravelPlanning, #DestinationGuides",
            "#ElonMusk #TechNews, Elon Musk, technology news, innovation, entrepreneurship, #ElonMusk, #TechnologyNews, #Innovation, #Entrepreneurship",
            "#GenerativeAI, artificial intelligence, generative models, creative AI, technology innovation, #ArtificialIntelligence, #GenerativeModels, #CreativeAI, #TechnologyInnovation",
            "Nvidia, GPU technology, AI computing, gaming hardware, technology, #Nvidia, #GPUTechnology, #AIComputing, #GamingHardware, #Technology",
            "Ukraine, geopolitics, international relations, conflict, global news, #Ukraine, #Geopolitics, #InternationalRelations, #Conflict, #GlobalNews",
            "google #gcf @google #gemini, Google, technology innovation, digital services, search engine, #Google, #TechnologyInnovation, #DigitalServices, #SearchEngine",
            "#JavaScript #ReactJS #GoLang #CloudComputing, programming languages, web development, cloud services, technology, #ProgrammingLanguages, #WebDevelopment, #CloudServices, #Technology",
            "#Serverless #DataScientist #Linux #Programming #Coding #100DaysofCode, cloud computing, data science, open-source, software development, #CloudComputing, #DataScience, #OpenSource, #SoftwareDevelopment",
            "#midjourney #image #ai, AI-generated imagery, creative technology, digital art, innovation, #AIGeneratedImagery, #CreativeTechnology, #DigitalArt, #Innovation",
            "#SpaceResearch #Astronomy, space exploration, astronomical research, cosmos, science, #SpaceExploration, #AstronomicalResearch, #Cosmos, #Science",
            "#RenewableEnergy #GreenTech, sustainable energy, environmental technology, clean energy, innovation, #SustainableEnergy, #EnvironmentalTechnology, #CleanEnergy, #Innovation",
            "#DigitalArt #ArtTech, digital creativity, art technology, new media art, innovation, #DigitalCreativity, #ArtTechnology, #NewMediaArt, #Innovation",
            "#FitnessTech #HealthInnovation, fitness innovation, health gadgets, wellness technology, personal health, #FitnessInnovation, #HealthGadgets, #WellnessTechnology, #PersonalHealth",
            "#EVs #HybridTech, electric vehicles, sustainable transportation, automotive technology, green vehicles, #ElectricVehicles, #SustainableTransportation, #AutomotiveTechnology, #GreenVehicles",
            "#ResourceManagementTech #Conservation, resource management, environmental conservation, sustainable practices, technology, #ResourceManagement, #EnvironmentalConservation, #SustainablePractices, #Technology",
            "#Drones #Innovation, drone technology, aerial innovation, unmanned aerial vehicles, technology, #DroneTechnology, #AerialInnovation, #UnmannedAerialVehicles, #Technology",
            "#CryptoNews #Cryptos #CryptoCommunity, cryptocurrency news, blockchain, digital finance, crypto community, #CryptocurrencyNews, #Blockchain, #DigitalFinance, #CryptoCommunity",
            "#CoinBase #Binance #BTC #ETH, cryptocurrency exchanges, Bitcoin, Ethereum, digital currency trading, #CryptocurrencyExchanges, #Bitcoin, #Ethereum, #DigitalCurrencyTrading",
            "#coffee #holiday #mood, coffee culture, holiday vibes, relaxation, lifestyle, #CoffeeCulture, #HolidayVibes, #Relaxation, #Lifestyle",
            "#economy, economic trends, financial markets, global economy, fiscal policy, #EconomicTrends, #FinancialMarkets, #GlobalEconomy, #FiscalPolicy",
            "#meme, internet culture, humor, social media trends, viral content, #InternetCulture, #Humor, #SocialMediaTrends, #ViralContent",
            "#indiegame #gamedevelopment, indie games, game design, game developers, creative gaming, #IndieGames, #GameDesign, #GameDevelopers, #CreativeGaming",
            "#gaming #game #dev #devs, video game development, gaming industry, game design, technology, #VideoGameDevelopment, #GamingIndustry, #GameDesign, #Technology",
            "#dotnet, .NET framework, software development, programming, Microsoft, #DotNet, #SoftwareDevelopment, #Programming, #Microsoft",
            "#Satoshi, Bitcoin, cryptocurrency founder, digital currency, blockchain technology, #Bitcoin, #CryptocurrencyFounder, #DigitalCurrency, #BlockchainTechnology",
            "#Muscle #AI #ArtificialIntelligence #Robotic #Robot #MachineLearning #Weight, AI in fitness, robotic assistance, machine learning applications, health technology, #AIInFitness, #RoboticAssistance, #MachineLearningApplications, #HealthTechnology",
            # "#tractors #drones #robotic, agricultural technology, drone farming, robotics in agriculture, innovation, #AgriculturalTechnology, #DroneFarming, #RoboticsInAgriculture, #Innovation",
            "#Robotics #AI #IoT, robotics technology, artificial intelligence, Internet of Things, automation, #RoboticsTechnology, #ArtificialIntelligence, #InternetOfThings, #Automation",
            "#movies #hollywood, cinema, Hollywood films, movie industry, entertainment, #Cinema, #HollywoodFilms, #MovieIndustry, #Entertainment"
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
