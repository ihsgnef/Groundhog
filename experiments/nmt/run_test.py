import os

cmd = 'python sample.py --source=fr.short --beam-search --beam-size 60 --trans trans.txt --state models/search_state_fr-en.pkl models/search_model_fr-en.npz 2>>log.txt'
# cmd = 'python sample.py --source=fr.short --beam-search --beam-size 60 --trans trans.txt --state search_state.pkl search_model.npz 2>>log.txt'

os.system(cmd)
