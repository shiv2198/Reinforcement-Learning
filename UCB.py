import numpy as np
import matplotlib.pyplot as plt
from ref_eps import start_game as st


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
        
    def ucb(m,n,nj):
        return m + np.sqrt(2*np.log(n) / nj)
        
        
            
def start_game():
            
    stand_means = [1,2,3]
    total_samples = 10000   
    total_plays = 0
        
    bandits = [Bandit(i) for i in stand_means]
    
    reward_log = [0]*total_samples
    
    for i in range(len(bandits)):
        value = bandits[i].pull_arm()
        total_plays += 1
        bandits[i].update_mean(value)
    
    for i in range(total_samples):
                    
        best_bound = np.argmax([Bandit.ucb(b.mean, total_plays, b.collected_samples) for b in bandits])
        value = bandits[best_bound].pull_arm()
        total_plays += 1
        bandits[best_bound].update_mean(value)
        
        reward_log[i] = value
            
    print([b.mean for b in bandits])
    plot(reward_log,total_samples,stand_means)




def plot(reward_log,total_samples,stand_means):
    cumulative_rewards = np.cumsum(reward_log)
    cumulative_average = cumulative_rewards / (np.arange(total_samples) + 1)
    #average = st()
    plt.plot(cumulative_average,label='UCB')
    #plt.plot(average,label='eps = 0.1')
    plt.plot(np.ones(total_samples)*stand_means[0])
    plt.plot(np.ones(total_samples)*stand_means[1])
    plt.plot(np.ones(total_samples)*stand_means[2])
    plt.xscale('log')
    plt.legend()
    plt.savefig('UCB.png')
    
      # linear plot
#    plt.plot(average, label='eps = 0.1')
#    plt.plot(cumulative_average, label='optimistic')
#    plt.legend()
#    plt.show()
#            
if __name__ == "__main__":
    start_game()
