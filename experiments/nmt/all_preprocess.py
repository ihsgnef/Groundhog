import os, sys
import state

src = state.source
dst = state.target
name = ''

if len(sys.argv) > 1:
    name = sys.argv[1]
    # name = name + '.'

cmd = 'perl web-demo/tokenizer.perl -l ' + src + ' < ' + name + src + ' > ' + name + src + '.tok'
os.system(cmd)

cmd = 'perl web-demo/tokenizer.perl -l ' + dst + ' < ' + name + dst + ' > ' + name + dst + '.tok'
os.system(cmd)


cmd = 'python preprocess/preprocess.py -d ' + name + src + '.vocab.pkl -v 30000 -b ' + name + src + '.binarized.pkl -p ' + name + src + '.tok'
os.system(cmd)

cmd = 'python preprocess/preprocess.py -d ' + name + dst + '.vocab.pkl -v 30000 -b ' + name + dst + '.binarized.pkl -p ' + name + dst + '.tok'
os.system(cmd)


cmd = 'python preprocess/invert-dict.py ' + name + src + '.vocab.pkl ' + name + src + '.ivocab.pkl'
os.system(cmd)

cmd = 'python preprocess/invert-dict.py ' + name + dst + '.vocab.pkl ' + name + dst + '.ivocab.pkl'
os.system(cmd)


cmd = 'python preprocess/convert-pkl2hdf5.py ' + name + src + '.binarized.pkl ' + name + src + '.binarized.h5'
os.system(cmd)

cmd = 'python preprocess/convert-pkl2hdf5.py ' + name + dst + '.binarized.pkl ' + name + dst + '.binarized.h5'
os.system(cmd)


cmd = 'python preprocess/shuffle-hdf5.py ' + name + src + '.binarized.h5 ' + name + dst + '.binarized.h5 ' + name + src + '.binarized.shuf.h5 ' + name + dst + '.binarized.shuf.h5'
os.system(cmd)
