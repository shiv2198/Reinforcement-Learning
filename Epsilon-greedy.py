import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    
    
    def __init__(self,win_rate):
        self.win_rate = win_rate
        self.mean = 0
        self.collected_samples = 0
        
    def pull_arm(self):
        return np.random.random() < self.win_rate
        
    def update_mean(self,value):
        self.collected_samples += 1
        self.mean = ( (self.collected_samples - 1)*self.mean + value ) / self.collected_samples
            
def start_game():
            
    win_rates = [0.25,0.5,0.60]
    total_samples = 10000
    Eps = 0.08
        
    bandits = [Bandit(i) for i in win_rates]
    
    reward_log = [0]*total_samples
    
    for i in range(total_samples):
    
    
        prob = np.random.random()
            
        if prob < Eps:
            
            number = np.random.randint(len(bandits))
                    
        else:
                    
            number = np.argmax([b.mean for b in bandits])
                    
        value = bandits[number].pull_arm()
        
        reward_log[i] = value
                
        bandits[number].update_mean(value)
            
    print([b.mean for b in bandits])
    plot(reward_log,total_samples,win_rates)

def plot(reward_log,total_samples,win_rates):
    cumulative_rewards = np.cumsum(reward_log)
    cumulative_average = cumulative_rewards / (np.arange(total_samples) + 1)
    plt.plot(cumulative_average)
    plt.plot(np.ones(total_samples)*win_rates[0])
    plt.plot(np.ones(total_samples)*win_rates[1])
    plt.plot(np.ones(total_samples)*win_rates[2])
    plt.xscale('log')
#    plt.plot(np.ones(total_samples)*np.max(win_rates))
    plt.savefig('greedy.png')
            
if __name__ == "__main__":
    start_game()