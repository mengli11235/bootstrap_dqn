# Bootstrap DQN with Noisy Priors

This repo is the implementation of the paper Improving the Diversity of Bootstrapped DQN via Noisy Priors on ALE games. 

# Requirements

atari-py installed from https://github.com/kastnerkyle/atari-py  
torch='1.0.1.post2'  
cv2='4.0.0'  
matplotlib='3.3.2'

# Usage
```run_bootstrap.py``` is the main file. ```visual_plot``` is the file to plot results.

To run experiment, set ```info``` in ```run_bootstrap.py```. To run different games, set game names in ```info['GAME']```. Set ```info['IMPROVEMENT']``` to use noisy priors or not.

```bash
python run_bootstrap.py
```

# Credit

Credit to which the project is built on: 

https://github.com/johannah/bootstrap_dqn
