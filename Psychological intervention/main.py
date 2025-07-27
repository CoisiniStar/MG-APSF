import openai
from world import World
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os
import random
import numpy as np
import json 
from scipy import optimize

def run_single_experiment(args, intervention_config, scenario_name):
    """运行单个实验场景"""
    print(f"\n🚀 Running {scenario_name} experiment...")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    checkpoint_path = f"checkpoint/{scenario_name}"
    output_path = f"output/{scenario_name}"
    
    # Create directories if they don't exist
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    # Create model with specific intervention configuration
    model = World(args, 
                  intervention_config=intervention_config,
                  initial_healthy=args.no_init_healthy,
                  initial_infected=args.no_init_infect,
                  contact_rate=args.contact_rate)
    
    # Run the model
    model.run_model(checkpoint_path, 0)
    
    # Extract data
    data = model.datacollector.get_model_vars_dataframe()
    df = pd.DataFrame(data)
    
    # Process infection data
    new_infections_newspaper = model.list_new_infected_cases[:-1]
    new_infections_newspaper[0] = model.list_new_infected_cases[0] + model.initial_infected
    new_infections_newspaper[1] = model.list_new_infected_cases[1] - model.initial_infected
    df['New Infections'] = new_infections_newspaper
    df['Cumulative Infections'] = df['New Infections'].cumsum()
    df['Total Contact'] = model.track_contact_rate[:len(df)]
    
    # Add metrics safely
    def safe_add_column(df, column_name, data_list, fill_value=np.nan):
        if len(data_list) == len(df):
            df[column_name] = data_list
        elif len(data_list) < len(df):
            padded_data = [fill_value] * (len(df) - len(data_list)) + data_list
            df[column_name] = padded_data
        else:
            df[column_name] = data_list[:len(df)]

    safe_add_column(df, 'Belief Average', model.belief_averages)
    safe_add_column(df, 'Belief Variance', model.belief_variances)  
    safe_add_column(df, 'Infection Rate', model.infection_rates)
    safe_add_column(df, 'Recovery Rate', model.recovery_rates)
    safe_add_column(df, 'TGI', model.tgi_values)

    df.insert(0, 'Step', range(0, len(df)))
    
    # Save data
    df.to_csv(f"{output_path}/{scenario_name}-data.csv")
    
    final_metrics = model.get_final_metrics()
    metrics_df = pd.DataFrame([final_metrics])
    metrics_df.to_csv(f"{output_path}/{scenario_name}-final_metrics.csv")
    
    # Save belief data
    save_belief_data(model, output_path, scenario_name)
    
    print(f"✅ {scenario_name} experiment completed")
    return {
        'scenario_name': scenario_name,
        'data': df,
        'model': model,
        'final_metrics': final_metrics,
        'intervention_config': intervention_config
    }

def save_belief_data(model, output_path, run_name):
    """保存belief数据到文件"""
    belief_timeline_data = []
    for agent_id, beliefs in model.agent_belief_timeline.items():
        for day_index, belief in enumerate(beliefs):
            belief_timeline_data.append({
                'agent_id': agent_id,
                'day': day_index,
                'belief': belief
            })
    
    if belief_timeline_data:
        df_timeline = pd.DataFrame(belief_timeline_data)
        df_timeline.to_csv(f'{output_path}/{run_name}-belief_timeline.csv', index=False)
    
    if model.belief_history:
        df_daily = pd.DataFrame(model.belief_history)
        df_daily.to_csv(f'{output_path}/{run_name}-belief_daily.csv', index=False)
    
    with open(f'{output_path}/{run_name}-belief_complete.json', 'w') as f:
        json.dump({
            'belief_history': model.belief_history,
            'agent_timeline': model.agent_belief_timeline
        }, f, indent=2)

def calculate_fitted_curve(infected_data):
    """计算拟合的感染曲线 (使用logistic growth model)"""
    def logistic_growth(t, K, r, t0):
        return K / (1 + np.exp(-r * (t - t0)))
    
    days = np.arange(len(infected_data))
    
    try:
        # Initial parameter guess
        K_guess = max(infected_data) * 1.2  # carrying capacity
        r_guess = 0.3  # growth rate
        t0_guess = len(infected_data) / 2  # inflection point
        
        popt, _ = optimize.curve_fit(
            logistic_growth, 
            days, 
            infected_data,
            p0=[K_guess, r_guess, t0_guess],
            maxfev=2000
        )
        
        fitted_curve = logistic_growth(days, *popt)
        return fitted_curve
    except:
        # Fallback: simple polynomial fit
        try:
            coeffs = np.polyfit(days, infected_data, 3)
            fitted_curve = np.polyval(coeffs, days)
            return np.maximum(fitted_curve, 0)  # Ensure non-negative
        except:
            return infected_data  # Return original data if fitting fails

def plot_population_dynamics(result, output_path):
    """绘制单个实验的人口动态变化图（类似你的第一张参考图）"""
    df = result['data']
    scenario_name = result['scenario_name']
    
    # 计算拟合曲线
    fitted_infected = calculate_fitted_curve(df['Infected'].values)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制主要曲线 - 匹配参考图的样式
    plt.plot(df['Step'], df['Infected'], 'b-o', linewidth=2, markersize=4, label='Infected')
    plt.plot(df['Step'], df['Susceptible'], color='orange', linewidth=2, marker='v', markersize=4, label='Susceptible')
    plt.plot(df['Step'], df['Recovered'], 'g-s', linewidth=2, markersize=4, label='Recovered')
    plt.plot(df['Step'], fitted_infected, 'r--', linewidth=2, alpha=0.8, label='Fitted Infected')
    
    # 设置图形样式
    plt.xlabel('Time (Days)', fontsize=12)
    plt.ylabel('Number of people', fontsize=12)
    plt.title(f'Population Dynamics - {scenario_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    plt.xlim(-0.5, len(df) - 0.5)
    plt.ylim(0, max(max(df['Infected']), max(df['Susceptible']), max(df['Recovered'])) * 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/population_dynamics_{scenario_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_intervention_comparison(all_results, output_path):
    """绘制干预策略对比图（1x3格式，参考你的图4样式）"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    scenarios = ['No Intervention', 'Official Intervention', 'Official + Psychological']
    
    for i, scenario in enumerate(scenarios):
        if scenario in all_results:
            df = all_results[scenario]['data']
            ax = axes[i]
            
            # 绘制三条主要曲线 - 匹配参考图样式
            ax.plot(df['Step'], df['Infected'], 'r-', linewidth=2, label='Infected')
            ax.plot(df['Step'], df['Susceptible'], 'b-', linewidth=2, label='Susceptible')  
            ax.plot(df['Step'], df['Recovered'], 'g-', linewidth=2, label='Recovered')
            
            # 设置子图样式
            ax.set_title(scenario, fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (Days)', fontsize=10)
            if i == 0:
                ax.set_ylabel('Number of people', fontsize=10)
            
            # 网格和图例
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            
            # 设置坐标轴范围
            ax.set_xlim(0, len(df) - 1)
            ax.set_ylim(0, max(max(df['Infected']), max(df['Susceptible']), max(df['Recovered'])) * 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/intervention_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_comparative_experiments(args):
    """运行所有对比实验"""
    # 定义三种干预策略
    intervention_scenarios = {
        'No Intervention': {
            'days': [], 
            'types': []
        },
        'Official Intervention': {
            'days': [6], 
            'types': ['official']
        },
        'Official + Psychological': {
            'days': [6, 7], 
            'types': ['official', 'psychological']
        }
    }
    
    all_results = {}
    
    # 运行每种场景的实验
    for scenario_name, intervention_config in intervention_scenarios.items():
        result = run_single_experiment(args, intervention_config, scenario_name.replace(' ', '_'))
        all_results[scenario_name] = result
    
    # 创建总输出目录
    main_output_path = "output/comparative_analysis"
    os.makedirs(main_output_path, exist_ok=True)
    
    # 绘制人口动态图（使用Official + Psychological的结果）
    if 'Official + Psychological' in all_results:
        plot_population_dynamics(all_results['Official + Psychological'], main_output_path)
    
    # 绘制干预策略对比图
    plot_intervention_comparison(all_results, main_output_path)
    
    # 生成汇总报告
    generate_summary_report(all_results, main_output_path)
    
    return all_results

def generate_summary_report(all_results, output_path):
    """生成汇总分析报告"""
    summary_data = []
    
    for scenario_name, result in all_results.items():
        metrics = result['final_metrics']
        summary_data.append({
            'Scenario': scenario_name,
            'Belief Average': metrics['belief_average'],
            'Belief Variance': metrics['belief_variance'],
            'Infection Rate': metrics['infection_rate'],
            'Recovery Rate': metrics['recovery_rate'],
            'Peak Rate': metrics['peak_rate'],
            'Half Rate': metrics['half_rate'],
            'TGI': metrics['tgi']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_path}/comparative_summary.csv', index=False)
    
    print("\n📊 Comparative Analysis Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GABM_Comparative", help="Name of the run to save outputs.")
    parser.add_argument("--contact_rate", default=3, type=int, help="Contact Rate")
    parser.add_argument("--no_init_healthy", default=28, type=int,
                        help="Number of initial healthy people in the world.")
    parser.add_argument("--no_init_infect", default=2, type=int,
                        help="Number of initial infected people in the world.")
    parser.add_argument("--no_days", default=15, type=int,
                        help="Total number of days the world should run.")
    parser.add_argument("--no_of_runs", default=1, type=int, help="Total number of times you want to run this code.")
    
    # Dataset arguments
    parser.add_argument("--train_path", default="datasets/Split_output_QID_train.json", help="Path to training dataset")
    parser.add_argument("--val_path", default="datasets/Split_output_QID_val.json", help="Path to validation dataset")
    parser.add_argument("--test_path", default="datasets/Split_output_QID_test.json", help="Path to test dataset")
    parser.add_argument("--num_news", default=50, type=int, help="Number of news to sample from datasets")
    parser.add_argument("--ensure_coverage", default=True, type=bool, help="Ensure coverage of all news types")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    
    # New argument for experiment type
    parser.add_argument("--experiment_type", default="comparative", 
                       choices=["single", "comparative"],
                       help="Type of experiment to run")
    
    args = parser.parse_args()
    print(f"Parameters: {args}")
    
    # Create necessary directories
    for dir_name in ["output", "checkpoint"]:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    # Set up OpenAI API
    openai.api_key = "sk-XXXXXXXXX"
    openai.api_base = "https://XXXXXX"

    if args.experiment_type == "comparative":
        # 运行对比实验
        print("🚀 Starting comparative intervention experiments...")
        all_results = run_comparative_experiments(args)
        print("✅ All comparative experiments completed!")
        
    else:
        # 运行单个实验（保持原有功能）
        print("🚀 Starting single experiment...")
        
        # Set random seed for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        checkpoint_path = f"checkpoint"
        output_path = f"output"
        
        # 默认使用双重干预策略
        intervention_config = {'days': [6, 7], 'types': ['official', 'psychological']}
        
        model = World(args, 
                      intervention_config=intervention_config,
                      initial_healthy=args.no_init_healthy,
                      initial_infected=args.no_init_infect,
                      contact_rate=args.contact_rate)
        
        model.run_model(checkpoint_path, 0)
        
        # 处理和保存数据（原有逻辑）
        data = model.datacollector.get_model_vars_dataframe()
        df = pd.DataFrame(data)
        
        new_infections_newspaper = model.list_new_infected_cases[:-1]
        new_infections_newspaper[0] = model.list_new_infected_cases[0] + model.initial_infected
        new_infections_newspaper[1] = model.list_new_infected_cases[1] - model.initial_infected
        df['New Infections'] = new_infections_newspaper
        df['Cumulative Infections'] = df['New Infections'].cumsum()
        df['Total Contact'] = model.track_contact_rate[:len(df)]
        
        def safe_add_column(df, column_name, data_list, fill_value=np.nan):
            if len(data_list) == len(df):
                df[column_name] = data_list
            elif len(data_list) < len(df):
                padded_data = [fill_value] * (len(df) - len(data_list)) + data_list
                df[column_name] = padded_data
            else:
                df[column_name] = data_list[:len(df)]

        safe_add_column(df, 'Belief Average', model.belief_averages)
        safe_add_column(df, 'Belief Variance', model.belief_variances)  
        safe_add_column(df, 'Infection Rate', model.infection_rates)
        safe_add_column(df, 'Recovery Rate', model.recovery_rates)
        safe_add_column(df, 'TGI', model.tgi_values)

        df.insert(0, 'Step', range(0, len(df)))
        df.to_csv(output_path + f"/{args.name}-data.csv")
        
        final_metrics = model.get_final_metrics()
        metrics_df = pd.DataFrame([final_metrics])
        metrics_df.to_csv(output_path + f"/{args.name}-final_metrics.csv")

        save_belief_data(model, output_path, args.name)
        
        print("\nFinal Metrics:")
        print(f"Belief Average: {final_metrics['belief_average']:.4f}")
        print(f"Belief Variance: {final_metrics['belief_variance']:.4f}")
        print(f"Infection Rate: {final_metrics['infection_rate']:.4f}")
        print(f"Recovery Rate: {final_metrics['recovery_rate']:.4f}")
        print(f"Peak Rate: {final_metrics['peak_rate']:.4f}")
        print(f"Half Rate: {final_metrics['half_rate']:.4f}")
        print(f"TGI: {final_metrics['tgi']:.4f}")

        # 原有的绘图代码
        plt.figure(figsize=(10, 6))
        plt.plot(df['Step'], df['Susceptible'], label="Susceptible")
        plt.plot(df['Step'], df['Infected'], label="Infected")
        plt.plot(df['Step'], df['Recovered'], label="Recovered")
        plt.xlabel('Step')
        plt.ylabel('# of People')
        plt.title('SIR Model with News Dataset')
        plt.legend()
        plt.savefig(output_path + f'/{args.name}-SIR.png', bbox_inches='tight')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(df['Step'], df['Belief Average'], label='Belief Average')
        axes[0, 0].set_title('Belief Average Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Belief Average')
        axes[0, 0].legend()
        
        axes[0, 1].plot(df['Step'], df['Belief Variance'], label='Belief Variance', color='orange')
        axes[0, 1].set_title('Belief Variance Over Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Belief Variance')
        axes[0, 1].legend()
        
        axes[1, 0].plot(df['Step'], df['Infection Rate'], label='Infection Rate', color='red')
        axes[1, 0].plot(df['Step'], df['Recovery Rate'], label='Recovery Rate', color='green')
        axes[1, 0].set_title('Infection and Recovery Rates')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Rate')
        axes[1, 0].legend()
        
        axes[1, 1].plot(df['Step'], df['TGI'], label='TGI', color='purple')
        axes[1, 1].set_title('Target Group Index (TGI)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('TGI')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path + f'/{args.name}-metrics.png', bbox_inches='tight')

        model.save_checkpoint(file_path=checkpoint_path + f"/{args.name}-completed.pkl")
        
        print("✅ Single experiment completed!")