from names_dataset import NameDataset
import numpy as np
import random
import openai
import time
import os
import shutil
import json

def probability_threshold(threshold):
    '''
    Used in self.infect_interaction()
    '''
    #Generates random number from 0 to 1
    
    return (np.random.rand()<threshold)

def load_datasets(train_path, val_path, test_path):
    '''
    Load and merge three JSON datasets
    '''
    all_news = []
    
    # Load train dataset
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        # Handle both list and single object formats
        if isinstance(train_data, list):
            all_news.extend(train_data)
        else:
            all_news.append(train_data)
    
    # Load val dataset
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
        if isinstance(val_data, list):
            all_news.extend(val_data)
        else:
            all_news.append(val_data)
    
    # Load test dataset
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        if isinstance(test_data, list):
            all_news.extend(test_data)
        else:
            all_news.append(test_data)
    
    return all_news

def sample_news_with_coverage(all_news, num_news, ensure_coverage=True):
    '''
    Sample news from all datasets ensuring coverage of all label types (0-5)
    '''
    if not ensure_coverage:
        return random.sample(all_news, min(num_news, len(all_news)))
    
    # Group news by label (convert string to int for processing)
    news_by_label = {}
    for news in all_news:
        label = int(news['label'])  # Convert string to int
        if label not in news_by_label:
            news_by_label[label] = []
        news_by_label[label].append(news)
    
    # 🆕 确保按标签0-5的顺序处理（可重现性）
    required_labels = list(range(6))  # [0, 1, 2, 3, 4, 5]
    
    # Ensure we have at least one news from each required label
    sampled_news = []
    for label in required_labels:
        if label in news_by_label and news_by_label[label]:
            sampled_news.append(random.choice(news_by_label[label]))
    
    # Fill remaining slots with random selection
    remaining_slots = num_news - len(sampled_news)
    if remaining_slots > 0:
        remaining_news = [n for n in all_news if n not in sampled_news]
        if remaining_news:
            sampled_news.extend(random.sample(remaining_news, min(remaining_slots, len(remaining_news))))
    
    return sampled_news

def mapped_label(label):
    '''
    Map label to official explanation with multi-granularity attribution (English version)
    
    Label mapping:
    0: Real News
    1: Image Fabrication
    2: Entity Inconsistency  
    3: Event Inconsistency
    4: Time & Space Inconsistency
    5: Ineffective Visual Information
    '''
    # Convert to int if it's a string
    if isinstance(label, str):
        label = int(label)
    
    LABEL_MAPPING = {
        0: "Through our multi-granularity attribution model analysis, this news content, images, and factual descriptions all conform to the actual situation. All information has passed multiple verification processes, therefore this news is determined to be real news.",
        
        1: "Through the multi-granularity attribution model, it is known that there are signs of manipulation in the pictures of this news, including image splicing, deep forgery, digital modification, or other forms of image tampering, therefore this news is determined to be fake news.",
        
        2: "Through the multi-granularity attribution model, it is known that the entity information mentioned in this news has inconsistencies. The descriptions of relevant people, organizations, institutions, or places do not match the actual situation or contain incorrect information, therefore this news is determined to be fake news.",
        
        3: "Through the multi-granularity attribution model, it is known that the events described in this news have inconsistencies. The occurrence process, results, impacts, or related details of the events do not match the actual situation or have been distorted in reporting, therefore this news is determined to be fake news.",
        
        4: "Through the multi-granularity attribution model, it is known that the time and space information in this news has inconsistencies. The time, location, spatiotemporal relationships, or timeline of the events do not match the actual situation, therefore this news is determined to be fake news.",
        
        5: "Through the multi-granularity attribution model, it is known that the visual information in this news is ineffective or misleading. Pictures, videos, or other visual elements do not match the news content, are irrelevant to the reported events, or are deceptive in nature, therefore this news is determined to be fake news."
    }
    
    return LABEL_MAPPING.get(label, "Unable to determine the authenticity of this news through our attribution model. We recommend treating this information with caution and seeking additional reliable sources for verification.")

def get_news_content(news_item):
    '''
    Extract clean content from news item
    '''
    content = news_item['content']
    
    # Remove URLs from content for cleaner text
    import re
    content_clean = re.sub(r'https?://\S+', '', content)
    content_clean = content_clean.strip()
    
    return content_clean

def calculate_belief_metrics(agents):
    '''
    Calculate belief-related metrics
    '''
    beliefs = [agent.beliefs[-1] for agent in agents]
    
    # Belief Average
    belief_avg = np.mean(beliefs)
    
    # Belief Variance
    belief_var = np.var(beliefs)
    
    return belief_avg, belief_var

def calculate_infection_rate(model):
    '''
    Calculate infection rate (new infections / susceptible)
    '''
    if model.susceptible == 0:
        return 0
    return model.daily_new_infected_cases / model.susceptible

def calculate_recovery_rate(model):
    '''
    Calculate recovery rate (new recoveries / infected)
    '''
    if model.infected == 0:
        return 0
    return model.daily_new_susceptible_cases / model.infected

def calculate_peak_rate(infection_history):
    '''
    Calculate peak infection rate
    '''
    if not infection_history:
        return 0
    return max(infection_history)

def calculate_half_rate(infection_history, peak_rate):
    '''
    Calculate half rate (time to reach half of peak)
    '''
    if peak_rate == 0:
        return 0
    half_peak = peak_rate / 2
    for i, rate in enumerate(infection_history):
        if rate >= half_peak:
            return i
    return 0

def calculate_tgi(agents, target_group_condition=None):
    '''
    Calculate Target Group Index (TGI)
    If target_group_condition is None, use infected agents as target group
    '''
    if target_group_condition is None:
        target_group = [agent for agent in agents if agent.health_condition == "Infected"]
    else:
        target_group = [agent for agent in agents if target_group_condition(agent)]
    
    if len(target_group) == 0:
        return 0
    
    total_population = len(agents)
    target_proportion = len(target_group) / total_population
    
    # TGI = (Target Group Proportion / Total Population Proportion) * 100
    # Since total population proportion is 1, TGI = target_proportion * 100
    tgi = target_proportion * 100
    
    return tgi

def generate_qualifications(n: int):
    '''
    Returns a list of random educational qualifications.

    Parameters:
    n (int): The number of qualifications to generate.
    '''

    # Define a list of possible qualifications including lower levels and no education
    qualifications = ['No Education', 'Primary School', 'Middle School',
                      'High School Diploma', 'Associate Degree', 'Bachelor\'s Degree', 
                      'Master\'s Degree', 'PhD', 'Professional Certificate']

    # Randomly select n qualifications from the list
    generated_qualifications = random.choices(qualifications, k=n)

    return generated_qualifications


def generate_names(n: int, s: int, country_alpha2='US'):
    '''
    Returns random names as names for agents from top names in the USA
    Used in World.init to initialize agents
    '''

    # This function will randomly selct n names (n/2 male and n/2 female) without
    # replacement from the s most popular names in the country defined by country_alpha2
    if n % 2 == 1:
        n += 1
    if s % 2 == 1:
        s += 1

    nd = NameDataset()
    male_names = nd.get_top_names(s//2, 'Male', country_alpha2)[country_alpha2]['M']
    female_names = nd.get_top_names(s//2, 'Female', country_alpha2)[country_alpha2]['F']
    if s < n:
        raise ValueError(f"Cannot generate {n} unique names from a list of {s} names.")
    # generate names without repetition
    names = random.sample(male_names, k=n//2) + random.sample(female_names, k=n//2)
    del male_names
    del female_names
    random.shuffle(names)
    return names


def generate_big5_traits(n: int):
    '''
    Return big 5 traits for each agent
    Used in World.init to initialize agents
    '''

    #Trait generation
    agreeableness_pos=['Cooperation','Amiability','Empathy','Leniency','Courtesy','Generosity','Flexibility',
                        'Modesty','Morality','Warmth','Earthiness','Naturalness']
    agreeableness_neg=['Belligerence','Overcriticalness','Bossiness','Rudeness','Cruelty','Pomposity','Irritability',
                        'Conceit','Stubbornness','Distrust','Selfishness','Callousness']
    #Did not use Surliness, Cunning, Predjudice,Unfriendliness,Volatility, Stinginess

    conscientiousness_pos=['Organization','Efficiency','Dependability','Precision','Persistence','Caution','Punctuality',
                            'Punctuality','Decisiveness','Dignity']
    #Did not use Predictability, Thrift, Conventionality, Logic
    conscientiousness_neg=['Disorganization','Negligence','Inconsistency','Forgetfulness','Recklessness','Aimlessness',
                            'Sloth','Indecisiveness','Frivolity','Nonconformity']

    surgency_pos=['Spirit','Gregariousness','Playfulness','Expressiveness','Spontaneity','Optimism','Candor'] 
    #Did not use Humor, Self-esteem, Courage, Animation, Assertion, Talkativeness, Energy level, Unrestraint
    surgency_neg=['Pessimism','Lethargy','Passivity','Unaggressiveness','Inhibition','Reserve','Aloofness'] 
    #Did not use Shyness, Silenece

    emotional_stability_pos=['Placidity','Independence']
    emotional_stability_neg=['Insecurity','Emotionality'] 
    #Did not use Fear, Instability, Envy, Gullibility, Intrusiveness
    
    intellect_pos=['Intellectuality','Depth','Insight','Intelligence'] 
    #Did not use Creativity, Curiousity, Sophistication
    intellect_neg=['Shallowness','Unimaginativeness','Imperceptiveness','Stupidity']


    #Combine each trait
    agreeableness_tot = agreeableness_pos + agreeableness_neg
    conscientiousness_tot = conscientiousness_pos + conscientiousness_neg
    surgency_tot = surgency_pos + surgency_neg
    emotional_stability_tot = emotional_stability_pos + emotional_stability_neg
    intellect_tot = intellect_pos + intellect_neg

    #create traits list to be returned
    traits_list = []

    for _ in range(n):
        agreeableness_rand = random.choice(agreeableness_tot)
        conscientiousness_rand = random.choice(conscientiousness_tot)
        surgency_rand = random.choice(surgency_tot)
        emotional_stability_rand = random.choice(emotional_stability_tot)
        intellect_rand = random.choice(intellect_tot)

        selected_traits=[agreeableness_rand,conscientiousness_rand,surgency_rand,
                                emotional_stability_rand,intellect_rand]

        traits_chosen = (', '.join(selected_traits))
        traits_list.append(traits_chosen)
    del agreeableness_rand
    del conscientiousness_rand
    del surgency_rand
    del emotional_stability_rand
    del intellect_rand
    del selected_traits
    del traits_chosen
    return traits_list


def update_day(agent):
    '''
    Update day funtion to update day_sick
    Used in World.step()
    '''
    # print("Agent ID: {} Day infected: {}".format(agent.unique_id,agent.day_infected))


    #if person is healthy, no reason to update health status
    if agent.health_condition=="Susceptible" or agent.health_condition=="Infected":
        return

    if agent.health_condition=="to_be_infected":
        agent.health_condition="Infected"
        agent.model.daily_new_infected_cases +=1 #every time new infection occurs in a day, counter is updated
        agent.model.infected += 1 #Update amount infected at any given time
        agent.model.susceptible -= 1

    if agent.health_condition=="to_be_recover":
        agent.health_condition="Recovered"
        agent.model.daily_new_susceptible_cases +=1 #every time new infection occurs in a day, counter is updated
        agent.model.infected -= 1 #Update amount infected at any given time
        agent.model.susceptible += 1


def factorize(n):
    '''
    Factorize number for ideal grid dimensions for # of agents
    Used in World.init
    '''
    for i in range(int(n**0.5), 1, -1):
        if n % i == 0:
            return (i, n // i)
    return (n, 1)

def get_completion_from_messages(messages, model="gpt-3.5-turbo-1106", temperature=0):
    success = False
    retry = 0
    max_retries = 30
    response = None  # 🆕 初始化response变量
    
    while retry < max_retries and not success:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature, # this is the degree of randomness of the model's output
            )
            success = True
        except Exception as e:
            print(f"Error: {e}\nRetrying...")
            retry += 1
            # 🆕 增加指数退避延迟
            delay = min(0.5 * (2 ** min(retry, 6)), 30)  # 最大30秒延迟
            time.sleep(delay)
    
    # 🆕 检查是否成功并提供兜底返回
    if not success or response is None:
        print(f"❌ All {max_retries} retries failed!")
        return "API调用失败，无法生成回复"
    
    return response.choices[0].message["content"]

def get_completion_from_messages_json(messages, model="gpt-3.5-turbo-1106", temperature=0):
    success = False
    retry = 0
    max_retries = 30
    response = None  # ✅ 你已经有了这个初始化
    
    while retry < max_retries and not success:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            
            # ✅ 你已经有了JSON验证
            content = response.choices[0].message["content"]
            json.loads(content)  # 测试是否为有效JSON
            success = True
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}\nRetrying...")
            retry += 1
            # 🆕 增加指数退避延迟
            delay = min(0.5 * (2 ** min(retry, 6)), 30)  # 最大30秒延迟
            time.sleep(delay)
        except Exception as e:
            print(f"API Error: {e}\nRetrying...")
            retry += 1
            # 🆕 增加指数退避延迟
            delay = min(0.5 * (2 ** min(retry, 6)), 30)  # 最大30秒延迟
            time.sleep(delay)
    
    # ✅ 你已经有了兜底返回，很完美！
    if not success or response is None:
        print("❌ All retries failed!")
        return '{"tweet": "API call failed", "belief": 0, "reasoning": "Failed to get response from API"}'
    
    return response.choices[0].message["content"]



def clear_cache():
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")