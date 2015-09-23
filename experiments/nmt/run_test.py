import os

# cmd = 'python score.py --mode=batch --src=test.en --trg=test.fr --allow-unk --state search_state.pkl search_model.npz >search_scores.txt 2>>log.txt'
cmd = 'python sample.py --source=test.fr --beam-search --beam-size 100 --trans search_trans.txt --state search_state.pkl search_model.npz 2>>log.txt'
#cmd = 'python sample1.py --source=test.fr --trans search_trans --state search_state.pkl search_model.npz 2>>log.txt'

os.system(cmd)
