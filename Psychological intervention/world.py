import mesa
import numpy as np
from citizen import Citizen
from tqdm import tqdm
from datetime import datetime, timedelta
from utils import (generate_names, generate_big5_traits, generate_qualifications, factorize, 
                   update_day, clear_cache, load_datasets, sample_news_with_coverage,
                   calculate_belief_metrics, calculate_infection_rate, calculate_recovery_rate,
                   calculate_peak_rate, calculate_half_rate, calculate_tgi, get_news_content)
import random
import pickle
from prompt import *

# functions for mesa.DataCollector in World class
def compute_num_susceptible(model):
    '''
    Computers number of susceptible agents for data frame
    '''
    return sum([1 for a in model.schedule.agents if a.health_condition == "Susceptible"])


def compute_num_infected(model):
    '''
    Computers number of infected agents for data frame
    '''
    return sum([1 for a in model.schedule.agents if a.health_condition == "Infected"])


def compute_num_recovered(model):
    '''
    Computers number of recovered agents for data frame
    '''
    return sum([1 for a in model.schedule.agents if a.health_condition == "Recovered"])


def compute_num_on_grid(model):
    '''
    Computers number of agents on the grid
    '''
    return sum([1 for a in model.schedule.agents if a.location == "grid"])


def compute_num_at_home(model):
    '''
    Computers number of agents at home
    '''
    return sum([1 for a in model.schedule.agents if a.location == "home"])


class World(mesa.Model):
    '''
    The world where Citizens exist
    '''
    def __init__(self, args, intervention_config=None, initial_healthy=2, initial_infected=1, contact_rate=5):
        
        ########################################
        #     Intialization of the world       #
        ########################################
        super().__init__()
        
        # 🆕 干预策略配置
        self.intervention_config = intervention_config or {'days': [6, 7], 'types': ['official', 'psychological']}
        
        #Agent initialization
        self.belief_history = []  # 记录每天所有agent的belief值
        self.agent_belief_timeline = {}  # 记录每个agent的belief时间线
        self.initial_healthy=initial_healthy
        self.initial_infected=initial_infected
        self.population=initial_healthy+initial_infected
        self.step_count = args.no_days
        self.offset = 0 #Offset for checkpoint load
        self.name = args.name
        
        # Load and sample news datasets
        self.all_news = load_datasets(
            args.train_path, 
            args.val_path, 
            args.test_path
        )
        self.sampled_news = sample_news_with_coverage(
            self.all_news, 
            args.num_news, 
            args.ensure_coverage
        )
        
        # Store news by label for easy access
        self.news_by_label = {}
        for news in self.sampled_news:
            label = news['label']
            if label not in self.news_by_label:
                self.news_by_label[label] = []
            self.news_by_label[label].append(news)

        #Important infection variables
        self.total_contact_rates = 0
        self.track_contact_rate = [0]
        self.list_new_infected_cases = [0]
        self.list_new_susceptible_cases = [0]
        self.daily_new_infected_cases = initial_infected
        self.daily_new_susceptible_cases = initial_healthy
        self.infected = initial_infected
        self.susceptible = initial_healthy
        self.current_date = datetime(2020, 3, 3)
        self.contact_rate= args.contact_rate   
        self.max_potential_interactions=0
        
        # Metrics tracking
        self.belief_averages = []
        self.belief_variances = []
        self.infection_rates = []
        self.recovery_rates = []
        self.tgi_values = []
        
        #Initialize Schedule
        self.schedule = mesa.time.RandomActivation(self)

        #Initiate data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={"Susceptible": compute_num_susceptible,
                            "Infected": compute_num_infected,
                            "Recovered": compute_num_recovered,
                            })
        

        ########################################
        #Assigning properties to all agents#
        ########################################

        #IDs for agents
        agent_id = 0 

        #generates list of random names out of the 200 most common names in the US
        names = generate_names(self.population, self.population*2)
        traits = generate_big5_traits(self.population)
        qualifications = generate_qualifications(self.population)

        #for loop to initialize each agents
        for i in range(self.population):
            #Creates healthy agents
            if i+1<=(self.initial_healthy):
                health_condition="Susceptible"
                # Select a random news item for susceptible agents
                news_item = random.choice(self.sampled_news)
                content = get_news_content(news_item)  # Use the new function to clean content
                opinion = f"I don't believe this news: {content}"
                topic = content

            #Creates infected, unhealthy agent(s)
            else:
                health_condition = "Infected"
                # Select a non-zero label news item for infected agents (assuming 0 is real news)
                fake_news = [n for n in self.sampled_news if int(n['label']) != 0]
                if fake_news:
                    news_item = random.choice(fake_news)
                else:
                    news_item = random.choice(self.sampled_news)
                content = get_news_content(news_item)  # Use the new function to clean content
                opinion = f"I believe this news: {content}"
                topic = content

            #create instances of the Citizen class
            citizen = Citizen(model=self,
                            unique_id=agent_id, name=names[i], age=random.randrange(18,65),
                            traits=traits[i], opinion=opinion,
                            qualification=qualifications[i],
                            health_condition=health_condition,
                            topic=topic
                            )
            # add agents to the scheduler
            self.schedule.add(citizen)
            # Updates to new agent ID
            agent_id += 1


    def decide_agent_interactions(self):
        '''
        Decides interaction partners for each agent
        '''
        self.max_potential_interactions = self.contact_rate
        for agent in self.schedule.agents:
            potential_interactions = [a for a in self.schedule.agents if a is not agent]
            random.shuffle(potential_interactions)
            potential_interactions=potential_interactions[:self.max_potential_interactions]
            for other_agent in potential_interactions:
                agent.agent_interaction.append(other_agent)

    def calculate_step_metrics(self):
        '''
        Calculate metrics for current step
        '''
        # Calculate belief metrics
        belief_avg, belief_var = calculate_belief_metrics(self.schedule.agents)
        self.belief_averages.append(belief_avg)
        self.belief_variances.append(belief_var)
        
        # Calculate infection and recovery rates
        infection_rate = calculate_infection_rate(self)
        recovery_rate = calculate_recovery_rate(self)
        self.infection_rates.append(infection_rate)
        self.recovery_rates.append(recovery_rate)
        
        # Calculate TGI
        tgi = calculate_tgi(self.schedule.agents)
        self.tgi_values.append(tgi)
    
    def collect_belief_data(self, current_day=None):
        """收集当前时间步所有agent的belief值"""
        actual_day = current_day if current_day is not None else self.schedule.time
        
        current_beliefs = {}
        daily_beliefs = []
        
        for agent in self.schedule.agents:
            if hasattr(agent, 'beliefs') and len(agent.beliefs) > 0:
                current_belief = agent.beliefs[-1]
                current_beliefs[agent.unique_id] = current_belief
                daily_beliefs.append(current_belief)
                
                if agent.unique_id not in self.agent_belief_timeline:
                    self.agent_belief_timeline[agent.unique_id] = []
                self.agent_belief_timeline[agent.unique_id].append(current_belief)
        
        # 使用正确的天数标记
        self.belief_history.append({
            'day': actual_day,  # 👈 使用真实天数
            'individual_beliefs': current_beliefs,
            'daily_average': sum(daily_beliefs) / len(daily_beliefs) if daily_beliefs else 0,
            'daily_variance': np.var(daily_beliefs) if daily_beliefs else 0
        })

    def step(self, current_day=None):
        '''
        Model time step
        '''

        # 使用传入的current_day而不是self.schedule.time
        actual_day = current_day if current_day is not None else self.schedule.time
        
        self.decide_agent_interactions()
    
        for agent in self.schedule.agents:
            self.total_contact_rates += len(agent.agent_interaction)
        self.track_contact_rate.append(self.total_contact_rates)
        self.total_contact_rates = 0

        # 🆕 参数化干预控制
        for agent in self.schedule.agents:
            if actual_day in self.intervention_config['days']:
                idx = self.intervention_config['days'].index(actual_day)
                intervention_type = self.intervention_config['types'][idx]
                
                if intervention_type == 'official':
                    agent.interact_official()
                elif intervention_type == 'psychological':
                    agent.interact_with_intervention()
                else:
                    agent.interact()
            else:
                agent.interact()

        for agent in self.schedule.agents:
            update_day(agent)
            
        # 🆕 添加这一行 - Mesa框架需要
        self.schedule.step()
        self.calculate_step_metrics()
        # 传入正确的天数
        self.collect_belief_data(current_day=actual_day)

    def get_final_metrics(self):
        '''
        Calculate final metrics including peak rate and half rate
        '''
        peak_rate = calculate_peak_rate(self.infection_rates)
        half_rate = calculate_half_rate(self.infection_rates, peak_rate)
        
        return {
            'belief_average': self.belief_averages[-1] if self.belief_averages else 0,
            'belief_variance': self.belief_variances[-1] if self.belief_variances else 0,
            'infection_rate': self.infection_rates[-1] if self.infection_rates else 0,
            'recovery_rate': self.recovery_rates[-1] if self.recovery_rates else 0,
            'peak_rate': peak_rate,
            'half_rate': half_rate,
            'tgi': self.tgi_values[-1] if self.tgi_values else 0
        }

    #Function to actually run the model
    def run_model(self, checkpoint_path, offset=0):
        self.offset = offset
        end_program=0
        for i in tqdm(range(self.offset,self.step_count)):
            print(f"📅 实际第{i}天，schedule.time={self.schedule.time}")
            #collect model level data
            self.datacollector.collect(self)

            # 模型步骤 - 传入真实的天数
            self.step(current_day=i)  # 👈 传入真实天数

            #collect all new cases from one day
            self.list_new_infected_cases.append(self.daily_new_infected_cases)
            self.list_new_susceptible_cases.append(self.daily_new_susceptible_cases)
            #set daily new case to 0 again
            self.daily_new_infected_cases = 0
            self.daily_new_susceptible_cases = 0

            #Print statements
            print(f"At the end of {self.current_date.date()}")
            print(f"Total Pop: {self.population}\t New Infected Cases: {self.list_new_infected_cases} \t New Susceptible_Cases: {self.list_new_susceptible_cases}")
            print (f"Currently Infected: {self.infected}")

            """
            early stopping condition: if there are no more infected agents left, 
            run for two more time steps, save the model and then end program
            """
            if self.infected==0:
                end_program+=1
            if end_program == 2:
                path = checkpoint_path + f"/{self.name}-final_early.pkl"
                self.save_checkpoint(file_path = path)
                break

            self.current_date += timedelta(days=1)
            path = checkpoint_path+f"/{self.name}-{i+1}.pkl"
            self.save_checkpoint(file_path = path)
            clear_cache()

    #saves checkpoint to specified file path
    def save_checkpoint(self, file_path):
        with open(file_path,"wb") as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load_checkpoint(file_path):
        with open(file_path,"rb") as file:
            return pickle.load(file)