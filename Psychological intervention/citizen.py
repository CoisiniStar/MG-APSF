# -- coding: utf-8 --
import pdb
import time
import mesa
import logging
from utils import get_completion_from_messages, get_completion_from_messages_json, probability_threshold, get_news_content, mapped_label
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
import json
from prompt import *

def get_official_statement(topic, news_label):
    """生成官方声明"""
    return mapped_label(news_label)

def get_summary_long(long_memory, short_memory):
    user_msg = long_memory_prompt.format(long_memory=long_memory, short_memory=short_memory)

    msg = [{"role": "user", "content": user_msg}]

    get_summary = get_completion_from_messages(msg, temperature=1)

    return get_summary

def get_summary_short(opinions,news_content):
    opinions_text = "\n".join(f"One people think: {opinion}" for opinion in opinions)

    user_msg = reflecting_prompt.format(opinions=opinions_text,news_content=news_content)

    msg = [{"role": "user", "content": user_msg}]

    get_summary = get_completion_from_messages(msg, temperature=0.5)

    return get_summary


class Citizen(mesa.Agent):
    '''
    Define who a citizen is:
    unique_id: assigns ID to agent
    name: name of the agent
    age: age of the agent
    traits: big 5 traits of the agent
    health_condition: flag to say if Susceptible or Infected or Recovered
    day_infected: agent attribute to count the number of days agent spends infected
    width, height: dimensions of world
    '''

    def __init__(self, model, unique_id, name, age, traits, qualification, health_condition, opinion, topic):
                
        super().__init__(unique_id,model) #Inherit mesa.Agent class attributes (model is mesa.Model)
        #Persona
        self.name = name
        self.age = age
        self.opinion=opinion
        self.traits=traits
        self.qualification=qualification
        self.topic=topic
        self.opinions = []
        self.beliefs = []
        self.long_opinion_memory = ''
        self.long_memory_full = []
        self.short_opinion_memory = []
        self.reasonings = []
        self.contact_ids = []

        #Health Initialization of Agent
        self.health_condition=health_condition

        #Contact Rate  
        self.agent_interaction=[]

        #Reasoning tracking
        self.persona = {"name":name, "age":age, "traits":traits}

        self.initial_belief, self.initial_reasoning = self.initial_opinion_belief()
        self.opinions.append(self.opinion)
        self.beliefs.append(self.initial_belief)
        self.reasonings.append(self.initial_reasoning)

    ########################################
    #          Initial Opinion             #
    ########################################
    def initial_opinion_belief(self):
        if self.health_condition == 'Infected':
            belief = 1
        else:
            belief = 0

        reasoning = 'initial_reasoning'

        return belief, reasoning

    ########################################
    #      Decision-helper functions       #
    ########################################


    ################################################################################
    #                       Meet_interact_infect functions                         #
    ################################################################################ 

    def interact(self):
        ''' 
        Step 1. Run infection for each agent_interaction
        Step 2. Reset agent_interaction for next day
        Used in self.step()
        '''
        
        others_opinions = []
        contact_id = []
        for agent in self.agent_interaction:
            contact_id.append(agent.unique_id)
            agent_latest_opinion = agent.opinions[-1]
            others_opinions.append(agent_latest_opinion)
        self.short_opinion_memory.append(others_opinions)
        self.contact_ids.append(contact_id)
        
        opinion_short_summary = get_summary_short(others_opinions, news_content=self.topic)

        long_mem = get_summary_long(self.long_opinion_memory, opinion_short_summary)

        user_msg = update_opinion_prompt.format(agent_persona=self.traits,
                                                agent_qualification=self.qualification,
                                                agent_name=self.name,
                                                long_mem=long_mem,
                                                news_content=self.topic,
                                                opinion=self.opinion)
        
        self.opinion, self.belief, self.reasoning = self.response_and_belief(user_msg)
        self.opinions.append(self.opinion)
        self.beliefs.append(self.belief)
        self.reasonings.append(self.reasoning)
        print(str(self.unique_id))
        print(self.reasoning)
        print(str(self.belief))

        self.long_opinion_memory = long_mem
        self.long_memory_full.append(self.long_opinion_memory)
        #Reset Agent Interaction list
        self.agent_interaction=[]
        self.get_health()

    def interact_official(self):
        ''' 
        Official interaction with news-specific explanations
        '''
        others_opinions = []
        contact_id = []
        for agent in self.agent_interaction:
            contact_id.append(agent.unique_id)
            agent_latest_opinion = agent.opinions[-1]
            others_opinions.append(agent_latest_opinion)

        # Get official statement based on the news content
        # Try to find the label for current topic
        news_label = "0"  # Default to real news (string format)
        for news in self.model.sampled_news:
            news_content = get_news_content(news)
            if news_content in self.topic:
                news_label = news['label']  # Keep as string
                break
        
        official_statement = get_official_statement(self.topic, news_label)
        others_opinions.append(official_statement)
        
        self.short_opinion_memory.append(others_opinions)
        self.contact_ids.append(contact_id)
        
        opinion_short_summary = get_summary_short(others_opinions, news_content=self.topic)

        long_mem = get_summary_long(self.long_opinion_memory, opinion_short_summary)

        user_msg = update_opinion_prompt.format(agent_persona=self.traits,
                                                agent_qualification=self.qualification,
                                                agent_name=self.name,
                                                long_mem=long_mem,
                                                news_content=self.topic,
                                                opinion=self.opinion)
        
        self.opinion, self.belief, self.reasoning = self.response_and_belief(user_msg)
        self.opinions.append(self.opinion)
        self.beliefs.append(self.belief)
        self.reasonings.append(self.reasoning)
        print(str(self.unique_id))
        print(self.reasoning)
        print(str(self.belief))

        self.long_opinion_memory = long_mem
        self.long_memory_full.append(self.long_opinion_memory)
        #Reset Agent Interaction list
        self.agent_interaction=[]
        self.get_health()
        
    ########################################
    #               Infect                 #
    ########################################
        
    def response_and_belief(self, user_msg):
        msg = [{"role": "user", "content": user_msg}]
        response_json = get_completion_from_messages_json(msg, temperature=1)
        
        try:
            output = json.loads(response_json)
            
            # 使用 .get() 提供默认值，双重保险
            tweet = output.get('tweet', 'No opinion generated')
            belief = output.get('belief', 0)
            reasoning = output.get('reasoning', 'No reasoning provided')
            
            belief = int(belief)
            return tweet, belief, reasoning
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Agent {self.unique_id} Parse error: {e}")
            return "Parse error", 0, "JSON parsing failed"

    def get_health(self):
        if self.health_condition=='Infected' and self.belief == 0:
            self.health_condition='to_be_recover'
        elif self.health_condition!='Infected' and self.belief == 1 :
            self.health_condition='to_be_infected'
        else:
            pass
    
    def interact_with_intervention(self):
        others_opinions = []
        contact_id = []
        for agent in self.agent_interaction:
            contact_id.append(agent.unique_id)
            agent_latest_opinion = agent.opinions[-1]
            others_opinions.append(agent_latest_opinion)
        self.short_opinion_memory.append(others_opinions)
        self.contact_ids.append(contact_id)
        
        opinion_short_summary = get_summary_short(others_opinions, news_content=self.topic)
        long_mem = get_summary_long(self.long_opinion_memory, opinion_short_summary)
        
        # Get official guidance for intervention - 修复标签获取逻辑
        news_label = "0"  # Default to real news (string format)
        for news in self.model.sampled_news:
            news_content = get_news_content(news)
            if news_content in self.topic:
                news_label = news['label']  # Keep as string
                break
        
        official_guidance = get_official_statement(self.topic, news_label)
        user_msg = psychological_intervention_prompt.format(
            agent_persona=self.traits,
            agent_qualification=self.qualification,
            agent_name=self.name,
            long_mem=long_mem,
            news_content=self.topic,
            opinion=self.opinion,
            official_guidance=official_guidance
        )
        
        self.opinion, self.belief, self.reasoning = self.response_and_belief(user_msg)
        self.opinions.append(self.opinion)
        self.beliefs.append(self.belief)
        self.reasonings.append(self.reasoning)
        print(str(self.unique_id))
        print(self.reasoning)
        print(str(self.belief))
        
        self.long_opinion_memory = long_mem
        self.long_memory_full.append(self.long_opinion_memory)
        self.agent_interaction = []
        self.get_health()

    ################################################################################
    #                              step functions                                  #
    ################################################################################ 
  
    def step(self):
        '''
        Step function for agent - 简化为只调用普通交互
        干预逻辑完全由world.py控制
        '''
        self.interact()