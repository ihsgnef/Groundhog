import os

# cmd = 'perl tokenizer.perl -l en < en > en.tok'
# os.system(cmd)
# cmd = 'perl tokenizer.perl -l fr < fr > fr.tok'
# os.system(cmd)

print 'vocab'
cmd = 'python preprocess.py -d en.vocab.pkl -v 30000 -b en.bin.pkl -p en.tok'
os.system(cmd)
cmd = 'python preprocess.py -d fr.vocab.pkl -v 30000 -b fr.bin.pkl -p fr.tok'
os.system(cmd)

print 'invert'
cmd = 'python invert-dict.py en.vocab.pkl en.ivocab.pkl'
os.system(cmd)
cmd = 'python invert-dict.py fr.vocab.pkl fr.ivocab.pkl'
os.system(cmd)

print 'convert h5'
cmd = 'python convert-pkl2hdf5.py en.bin.pkl en.bin.h5'
os.system(cmd)
cmd = 'python convert-pkl2hdf5.py fr.bin.pkl fr.bin.h5'
os.system(cmd)

print 'shuffle'
cmd = 'python shuffle-hdf5.py en.bin.h5 fr.bin.h5 en.shuf.h5 fr.shuf.h5'
os.system(cmd)
