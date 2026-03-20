# MG-APSF: From Prediction to Intervention in Fake News Dissemination 🚀

> Official implementation for the rumor-debunking simulation framework based on Psychological Intervention and Multi-Granularity Attribution Reasoning.

## 📖 Overview

In the digital era, the rapid propagation of fake news through social media platforms brings a significant challenge to establishing a clear and healthy online ecosystem. Existing methods only rely on modeling semantic consistency to make predictions. However, these methods are unable to attribute different types of fake news and largely ignore the public's belief in refuting information. 

To address these limitations, we propose a novel **M**ulti-**G**ranularity **A**ttribution and **P**sychological intervention **S**imulation **F**ramework for rumor-debunking, dubbed ***MG-APSF***. This innovative simulation framework combines SLMs and LLMs, specifically designed to enhance the public's trust in rumor-refuting information. 

Our experimental results demonstrate that MG-APSF can not only attribute different types of fake news but also significantly improve public trust in refuting information and suppress the spread of rumors.

## ✨ Key Features

* **Multi-Granularity Attribution Reasoning**: Utilizes small language models (SLMs) with acute insights as multi-granularity spotters to uncover the disguise of fake news at varying levels (Image Manipulation, Fact-checking, Event Consistency, Spatio-Temporal Matching).
* **LLM-based Multi-Agent Simulation**: Built on the `mesa` framework, modeling a dynamic social network where agents with distinct personalities interact, propagate information, and update their beliefs based on the **SIR (Susceptible-Infected-Recovered)** epidemic model.
* **Dual Intervention Mechanism**: Introduces Official Agents (providing fine-grained factual refutation) and Psychological Intervention Agents (addressing individual confirmation bias) to effectively halt the spread of disinformation.

## 🧬 Multi-Agent Simulation Engine (Powered by Mesa)

The core of our social simulation is implemented in Python using the `mesa` agent-based modeling framework and powered by LLMs (e.g., OpenAI GPT series) for cognitive reasoning.

### 🌍 World Environment (`world.py`)
The environment acts as the macro-scheduler for the simulation:
* **Dynamic Social Network**: Manages agent interactions based on a predefined contact rate.
* **Intervention Scheduling**: Precisely controls the timing of interventions (e.g., Official refutation on Day 6, Psychological intervention on Day 7).
* **Metrics Tracking**: Utilizes `mesa.DataCollector` to track global metrics including *Belief Average*, *Belief Variance*, *Infection/Recovery Rates*, and *Target Group Index (TGI)*.

### 👤 Citizen Agents (`citizen.py`)
Each `Citizen` inherits from `mesa.Agent` and simulates a realistic human participant:
* **Persona Diversity**: Agents are instantiated with varying ages, educational qualifications, and **Big Five Personality Traits** (Neuroticism, Extraversion, Agreeableness, etc.).
* **SIR Health States**: Agents transition between `Susceptible` (vulnerable to fake news), `Infected` (believing and spreading fake news), and `Recovered` (accepting the truth).
* **Memory Mechanism**: Agents maintain both **Short-term Memory** (recent daily interactions) and **Long-term Memory** (accumulated beliefs and reflections), integrated dynamically via LLM prompts.
* **Confirmation Bias Modeling**: During psychological interventions, the LLM is explicitly prompted to simulate human confirmation bias, where agents may question official sources that contradict their pre-existing beliefs (`prompt.py`).

## 📁 Repository Structure

```text
├── checkpoint/             # Directory for saving/loading simulation states
├── datasets/               # AMG & PHEME datasets (JSON format)
├── output/                 # Generated CSV metrics and visualization plots
├── analysis.py             # Script to extract and analyze agent memory traces
├── check_dataset.py        # Utility to verify dataset integrity and label distribution
├── citizen.py              # LLM-based Agent definitions (Mesa Agent)
├── main.py                 # Entry point for running simulations (Single/Comparative)
├── prompt.py               # Prompt templates for memory, reflection, and interventions
├── utils.py                # Helper functions for metrics, personas, and API calling
├── world.py                # Simulation Environment definition (Mesa Model)
└── requirements.txt        # Package dependencies
```

# 🛠️ Installation & Setup
1. Clone the repository:

```bash
     git clone [https://github.com/yourusername/MG-APSF.git](https://github.com/yourusername/MG-APSF.git)
     cd MG-APSF
```

2. Install dependencies:
```bash
     pip install -r requirements.txt
```

3. Configure API Keys:
Open 'main.py' and set your OpenAI API key and Base URL:
     ```bash
     openai.api_key = "sk-YOUR_API_KEY"
     openai.api_base = "https://YOUR_BASE_URL"
     ```
# 🚀 Usage
The framework supports running both single-scenario simulations and comprehensive comparative experiments.

Run a Comparative Experiment (Recommended):
This will run the simulation under three conditions: No Intervention, Official Intervention, and Official + Psychological Intervention, and automatically generate comparative plots.

     ```bash
          python main.py --experiment_type comparative --no_days 15 --contact_rate 3
     ```

Run a Single Custom Experiment:
     ```bash
          python main.py --experiment_type single --name "My_Custom_Run" --no_days 15 --no_init_healthy 28 --no_init_infect 2
     ```
# 📊 Outputs
After execution, check the output/ folder for:

\item population_dynamics_*.png: SIR model state transitions over time.

\item intervention_comparison.png: Side-by-side comparison of different intervention strategies.

\item *-data.csv & *-final_metrics.csv: Raw metric data for further analysis.

# 📝 Ethical Considerations
This framework relies on large language models and is intended solely for academic research purposes to understand and mitigate the spread of misinformation. It strictly prohibits the use of these techniques for the creation or dissemination of malicious content.
