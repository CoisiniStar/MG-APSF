from world import World

def write_txt(list, name):
    with open(name, "w", encoding="utf-8") as file:
        for item in list:
            file.write(str(item)+'\n')

#Load any checkpoint
model=World.load_checkpoint(r"checkpoint\run-1\GABM-1.pkl")

long_memory = []
long_memory_full = []
short_memory = []
reasoning = []
#Get the responses and other relevant attributes of the agents over time
for agent in model.schedule.agents:
    long_memory.append(agent.long_opinion_memory)
    short_memory.append(agent.short_opinion_memory)
    long_memory_full.append(agent.long_memory_full)
    reasoning.append(agent.reasonings)

write_txt(long_memory, 'long.txt')
write_txt(long_memory_full, 'long_full.txt')
write_txt(short_memory, 'short.txt')
write_txt(reasoning, 'reasoning.txt')
