import numpy as np
import matplotlib.pyplot as plt
from Epsilon_greedy import start_game as st


class Bandit:
    
    
    def __init__(self,v):
        self.v = v
        self.mean = 10
        self.collected_samples = 0
        
    def pull_arm(self):
        return np.random.randn() + self.v
        
    def update_mean(self,value):
        self.collected_samples += 1
        self.mean = (1-1/self.collected_samples)*self.mean+1/self.collected_samples*value
            
def start_game():
            
    stand_means = [1,2,3]
    total_samples = 10000   

        
    bandits = [Bandit(i) for i in stand_means]
    
    reward_log = [0]*total_samples
    
    for i in range(total_samples):
                    
        number = np.argmax([b.mean for b in bandits])
        value = bandits[number].pull_arm()
        
        reward_log[i] = value
                
        bandits[number].update_mean(value)
            
    print([b.mean for b in bandits])
    plot(reward_log,total_samples,stand_means)

def plot(reward_log,total_samples,stand_means):
    cumulative_rewards = np.cumsum(reward_log)
    cumulative_average = cumulative_rewards / (np.arange(total_samples) + 1)
    average = st()
    plt.plot(cumulative_average,label='OIV')
    plt.plot(average,label='eps = 0.1')
    plt.plot(np.ones(total_samples)*stand_means[0])
    plt.plot(np.ones(total_samples)*stand_means[1])
    plt.plot(np.ones(total_samples)*stand_means[2])
    plt.xscale('log')
    plt.legend()
    plt.savefig('OIV_vs_Greedy.png')
    
      # linear plot
#    plt.plot(average, label='eps = 0.1')
#    plt.plot(cumulative_average, label='optimistic')
#    plt.legend()
#    plt.show()
#            
if __name__ == "__main__":
    start_game()
