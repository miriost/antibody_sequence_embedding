#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 14:32:57 2018

original @author: https://github.com/kyu999/biovec/blob/master/biovec/models.py
@author: miri-o

"""

from gensim.models import word2vec
from Bio import SeqIO
from random import randint


def split_ngrams_with_repetition(seq, n):
    """
    'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
    """
    a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
    str_ngrams = []
    for ngrams in [a,b,c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams

def split_ngrams_no_repetition(seq, n, reading_frame):
    """
    'acccgtgtctgg', n=3, reading frame = 1: ['acc', 'cgt', 'gtc', 'tgg']
    reading frame = 2: ['ccc', 'gtg', 'tct']
    reading frame = 3: ['ccg', 'tgt', 'ctg']
    """
    a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
    str_ngrams = []
    for ngrams in [a,b,c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams[reading_frame-1]

def split_to_random_n_grams(seq, n):
    """
    n = a tuple (start, end) indicating the range of n's for random sampling 
    e.g. n = (3, 8) will split the sequence to n-grams with sizes ranging from 3-8
    'AGAMQSASMRDSRGPDPVSRATHNWFDP' : ['AGA', 'MQSASM', 'RDSRGPDPV', 'SRAT', 'HNWFDP']
    """
    str_ngrams = []
    current_n = randint(n[0], n[1])
    while len(seq)-n[0]>current_n:
        str_ngrams.append(str(seq[:current_n]))
        seq = seq[current_n:]
        current_n = randint(n[0], n[1])
    str_ngrams.append(str(seq))
    return str_ngrams
    
def generate_corpusfile(corpus_fname, n, out, reading_frame = None, trim = None):
    '''
    Args:
        corpus_fname: corpus file name
        n: the number of chunks to split. In other words, "n" for "n-gram". 
        for a constant n splitting - n is an integer, for a random range, n should be a tuple of (start, end)
        reading_frame: 1/2/3 for splitting, default: None, including repetition (generating 3 overlaps)
        out: output corpus file path
        trim : typle (from start, drom end) - how many characteres to trim from the beginning and from the end
    Description:
        Protvec uses word2vec inside, and it requires to load corpus file
        to generate corpus.
        
    '''
    f = open(out, "w")
    
    for r in SeqIO.parse(corpus_fname, "fasta"):
        if trim:
            r.seq = r.seq[trim[0]:-trim[1]]
        if (reading_frame == None) and type(n) == int:
            ngram_patterns = split_ngrams_with_repetition(r.seq, n)
        elif type(n) == int:
            ngram_patterns = split_ngrams_no_repetition(r.seq, n, reading_frame)
        elif type(n) == tuple:
            ngram_patterns = split_to_random_n_grams(r.seq, n)
        else:
            print('Error building corpus file, make sure n is and integer for contant n-grams, or a tuple for random length n-grams')
            f.close()
            break
        if type(ngram_patterns[0])==list:
            for sub_seq in ngram_patterns:
                f.write(" ".join(sub_seq) + "\n")
        else:
            f.write(" ".join(ngram_patterns) + "\n")
    f.close()

def load_protvec(model_fname):
    return word2vec.Word2Vec.load(model_fname)


class ProtVec(word2vec.Word2Vec):

    def __init__(self, corpus_fname=None, corpus=None, n=3, reading_frame = None, trim = None, size=100, out="corpus.txt",  sg=1, window=25, min_count=2, workers=3):
        """
        Either fname or corpus is required.
        corpus_fname: fasta file for corpus
        corpus: corpus object implemented by gensim
        n: n of n-grams. single integer for a costant n, and a string ‘(start, end)’ for random splitting.
        reading frame : default None. possible values: 1/2/3/None – for all options
        trim: paramter for trimming the sequences, string format ‘(chars from start, chars from end)’ 
        out: corpus output file path
        min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
        """

        self.n = n
        self.reading_frame = reading_frame
        self.size = size
        self.corpus_fname = corpus_fname
        self.trim = trim
        self.window = window

        if corpus is None and corpus_fname is None:
            raise Exception("Either corpus_fname or corpus is needed!")

        if corpus_fname is not None:
            print('Generate Corpus file from fasta file...')
            generate_corpusfile(corpus_fname, n, out + '_corpus.txt', reading_frame, trim)
            corpus = word2vec.Text8Corpus(out + '_corpus.txt')
            

        word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count, workers=workers)
        print('word2vec model, size={}, window={}, min_count={}, workers={})'.format(size, window, min_count, workers))

    def to_vecs(self, seq):
        """
        convert sequence to three n-length vectors
        e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]
        """
        ngram_patterns = split_ngrams_with_repetition(seq, self.n)

        protvecs = []
        for ngrams in ngram_patterns:
            ngram_vecs = []
            for ngram in ngrams:
                try:
                    ngram_vecs.append(self[ngram])
                except:
                    raise KeyError("Model has never trained this n-gram: " + ngram)
            protvecs.append(sum(ngram_vecs))
            return protvecs
        
        
class AAVec(word2vec.Word2Vec):

    def __init__(self, corpus_fname=None, corpus=None, n=3, reading_frame=1, trim = None, size=100, out="corpus.txt",  sg=1, window=25, min_count=2, workers=3):
        """
        Either fname or corpus is required.
        corpus_fname: fasta file for corpus
        corpus: corpus object implemented by gensim
        n: n of n-gram
        out: corpus output file path
        min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
        """

        self.n = n
        self.size = size
        self.corpus_fname = corpus_fname
        self.reading_frame = reading_frame
        self.trim = trim

        if corpus is None and corpus_fname is None:
            raise Exception("Either corpus_fname or corpus is needed!")

        if corpus_fname is not None:
            print('Generate Corpus file from fasta file...')
            generate_corpusfile(corpus_fname, n, out, reading_frame)
            corpus = word2vec.Text8Corpus(out)
            

        word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count, workers=workers)
        print('word2vec model, size={}, window={}, min_count={}, workers={})'.format(size, window, min_count, workers))

    def to_vecs(self, seq):
        """
        convert sequence to n-length vector
        e.g. 'AGAMQSASM' => [ array([  ... * 100 ])]
        """
        ngrams = split_ngrams_no_repetition(seq, self.n, self.reading_frame)

        protvecs = []
        ngram_vecs = []
        for ngram in ngrams:
            try:
                ngram_vecs.append(self[ngram])
            except:
                raise Exception("Model has never trained this n-gram: " + ngram)
        protvecs.append(sum(ngram_vecs))
        return protvecs