import os, sys, subprocess, uuid
import state

src = state.source
trg = state.target
train_dir = state.datadir
valid_dir = state.evaldir
src_file = train_dir + src
trg_file = train_dir + trg
src_token = train_dir + src + '.tok'
trg_token = train_dir + trg + '.tok'
src_vocab = train_dir + src + '.vocab.pkl'
trg_vocab = train_dir + trg + '.vocab.pkl'
src_ivocab = train_dir + src + '.ivocab.pkl'
trg_ivocab = train_dir + trg + '.ivocab.pkl'
merged = train_dir + 'merged'
shuffled = train_dir + 'shuffled'

tokenizer = os.path.join('preprocess', 'tokenizer.perl')
preprocessor = os.path.join('preprocess', 'preprocess.py')
dict_inverter = os.path.join('preprocess', 'invert-dict.py')

def tokenize(files, tokenizer):
    for fname in files:
        outfile = fname + '.tok'
        var = ['perl', tokenizer, '-l', fname.split('/')[-1].split('.')[0]]
        if not os.path.exists(outfile):
            with open(fname, 'r') as inp:
                with open(outfile, 'w', 0) as out:
                    subprocess.Popen(var, stdin=inp, stdout=out, shell=False)

def vocabularize(preprocessor, inverter):
    if not os.path.exists(src_vocab):
        subprocess.call(' python {} -d {} -v {} {}'.format(
            preprocessor, src_vocab, 30000, src_token), shell=True)
    if not os.path.exists(trg_vocab):
        subprocess.call(' python {} -d {} -v {} {}'.format(
            preprocessor, trg_vocab, 30000, trg_token), shell=True)
    if not os.path.exists(src_ivocab):
        subprocess.call(' python {} {} {}'.format(
            inverter, src_vocab, src_ivocab), shell=True)
    if not os.path.exists(trg_ivocab):
        subprocess.call(' python {} {} {}'.format(
            inverter, trg_vocab, trg_ivocab), shell=True)


def merge_parallel(src_filename, trg_filename, merged_filename):
    with open(src_filename, 'r') as left:
        with open(trg_filename, 'r') as right:
            with open(merged_filename, 'w') as final:
                while True:
                    lline = left.readline()
                    rline = right.readline()
                    if (lline == '') or (rline == ''):
                        break
                    if (lline != '\n') and (rline != '\n'):
                        final.write(lline[:-1] + ' ||| ' + rline)

def split_parallel(merged_filename, src_filename, trg_filename):
    with open(merged_filename, 'r') as combined:
        with open(src_filename, 'w') as left:
            with open(trg_filename, 'w') as right:
                for line in combined:
                    line = line.split('|||')
                    left.write(line[0].strip() + '\n')
                    right.write(line[1].strip() + '\n')

        
def shuffle_parallel():
    src_out = src_file + '.shuf'
    trg_out = trg_file + '.shuf'
    shuffled_file = str(uuid.uuid4())
    if not os.path.exists(src_out) or not os.path.exists(trg_out):
        merge_parallel(src_file, trg_file, merged)
        subprocess.call(' shuf {} > {}'.format(merged, shuffled), shell=True)
        split_parallel(shuffled, src_out, trg_out)
    '''
    if os.path.exists(merged_file):
        os.remove(merged_file)
    if os.path.exists(shuffled_file):
        os.remove(shuffled_file)
    '''



def main():
    train_files = [os.path.join(train_dir, src), os.path.join(train_dir, trg)]
    valid_files = [os.path.join(valid_dir, src), os.path.join(valid_dir, trg)]
    tokenize(train_files + valid_files, tokenizer)
    vocabularize(preprocessor, dict_inverter)
    shuffle_parallel()

if __name__ == '__main__':
    main()
