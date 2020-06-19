#
## https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
## https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
#from __future__ import print_function, division
#from builtins import range
## Note: you may need to update your version of future
## sudo pip install -U future
#
#import numpy as np
#import matplotlib.pyplot as plt
#from Epsilon_greedy import start_game
#
#
#class Bandit:
#  def __init__(self, m, upper_limit):
#    self.m = m
#    self.mean = upper_limit
#    self.N = 1
#
#  def pull(self):
#    return np.random.randn() + self.m
#
#  def update(self, x):
#    self.N += 1
#    self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N*x
#
#
#def run_experiment(m1, m2, m3, N, upper_limit=10):
#  bandits = [Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit)]
#
#  data = np.empty(N)
#  
#  for i in range(N):
#    # optimistic initial values
#    j = np.argmax([b.mean for b in bandits])
#    x = bandits[j].pull()
#    bandits[j].update(x)
#
#    # for the plot
#    data[i] = x
#  cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
#
#  # plot moving average ctr
#  plt.plot(cumulative_average)
#  plt.plot(np.ones(N)*m1)
#  plt.plot(np.ones(N)*m2)
#  plt.plot(np.ones(N)*m3)
#  plt.xscale('log')
#  plt.show()
#
#  for b in bandits:
#    print(b.mean)
#
#  return cumulative_average
#
#if __name__ == '__main__':
#  c_1 = start_game()
#  oiv = run_experiment(1.0, 2.0, 3.0, 1000)
#
#  # log scale plot
#  plt.plot(c_1, label='eps = 0.1')
#  plt.plot(oiv, label='optimistic')
#  plt.legend()
#  plt.xscale('log')
#  plt.show()
#
#
#  # linear plot
##  plt.plot(c_1, label='eps = 0.1')
#  plt.plot(oiv, label='optimistic')
#  plt.legend()
#  plt.show()


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
    plt.savefig('OIV_vs_Greedy#.png')
    
      # linear plot
#    plt.plot(average, label='eps = 0.1')
#    plt.plot(cumulative_average, label='optimistic')
#    plt.legend()
#    plt.show()
#            
if __name__ == "__main__":
    start_game()