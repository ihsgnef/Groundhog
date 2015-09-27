import os

cmd = 'python sample.py --source=en.short --beam-search --beam-size 60 --trans trans.txt --state search_state_fr-en.pkl search_model_fr-en.npz 2>>log.txt'

os.system(cmd)
