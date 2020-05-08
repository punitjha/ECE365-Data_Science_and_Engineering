from nose.tools import eq_
from ece365lib import train
import nose
import nltk
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from ece365lib import evaluate

def setup_module():
    global food_corpus, natr_corpus
    
    food = ['barley', 'castor-oil', 'cocoa', 'coconut', 'coconut-oil', 'coffee', 'copra-cake''grain', 'groundnut', 'groundnut-oil', 'potato''soy-meal', 'soy-oil', 'soybean', 'sugar', 'sun-meal', 'sun-oil', 'sunseed', 'tea', 'veg-oil', 'wheat']
    natural_resources = ['alum', 'fuel', 'gas', 'gold', 'iron-steel', 'lead', 'nat-gas', 'palladium', 'propane', 'tin', 'zinc']

    corpus = nltk.corpus.reuters
    food_corpus = corpus.raw(categories=food)
    natr_corpus = corpus.raw(categories=natural_resources)
    
    
def test_d1_1_tk():
    global food_corpus, natr_corpus
    
    food_corpus_tk = train.tokenize_corpus(food_corpus)
    natr_corpus_tk = train.tokenize_corpus(natr_corpus)
    
    eq_(food_corpus_tk[25][5],'Monday')
    eq_(natr_corpus_tk[25][5],'are')
    
    
def test_d1_2_pad():
    global food_corpus, natr_corpus
    
    food_corpus_tk = train.tokenize_corpus(food_corpus)
    natr_corpus_tk = train.tokenize_corpus(natr_corpus)
    
    food_corpus_tk_pd = train.pad_corpus(food_corpus_tk)
    natr_corpus_tk_pd = train.pad_corpus(natr_corpus_tk)
    
    eq_(food_corpus_tk_pd[35][0], '<s>')
    eq_(natr_corpus_tk_pd[35][-1], '</s>')
    eq_(len(food_corpus_tk_pd[45]), 14)
    eq_(len(natr_corpus_tk_pd[45]), 19)
    eq_(len(food_corpus_tk_pd[45]) - len(food_corpus_tk[45]), 2)
    
    
def test_d1_3_spc():
    global food_corpus, natr_corpus
    
    food_corpus_tk = train.tokenize_corpus(food_corpus)
    natr_corpus_tk = train.tokenize_corpus(natr_corpus)
    
    food_corpus_tk_pd = train.pad_corpus(food_corpus_tk)
    natr_corpus_tk_pd = train.pad_corpus(natr_corpus_tk)
    
    food_corpus_tr, food_corpus_te = train.split_corpus(food_corpus_tk_pd)
    natr_corpus_tr, natr_corpus_te = train.split_corpus(natr_corpus_tk_pd)
    
    eq_(len(food_corpus_tr), 4888)
    eq_(len(food_corpus_te), 1222)
    eq_(len(natr_corpus_tr), 2610)
    eq_(len(natr_corpus_te), 653)
    eq_(food_corpus_te[3][5], 'by')
    eq_(natr_corpus_te[1][2], 'Project')
    
    
def test_d1_4_cn():
    global food_corpus, natr_corpus
    
    food_corpus_tk = train.tokenize_corpus(food_corpus)
    natr_corpus_tk = train.tokenize_corpus(natr_corpus)
    
    food_corpus_tk_pd = train.pad_corpus(food_corpus_tk)
    natr_corpus_tk_pd = train.pad_corpus(natr_corpus_tk)
    
    food_corpus_tr, food_corpus_te = train.split_corpus(food_corpus_tk_pd)
    natr_corpus_tr, natr_corpus_te = train.split_corpus(natr_corpus_tk_pd)
    
    food_ngrams, food_vocab_man = train.count_ngrams(food_corpus_tr, 3)
    natr_ngrams, natr_vocab_man = train.count_ngrams(natr_corpus_tr, 3)
    
    eq_(len(food_ngrams.keys()), 181387)
    eq_(len(natr_ngrams.keys()), 105612)
    eq_(food_ngrams[('sold', 'the')], 2)
    eq_(natr_ngrams[('extracting', 'the')], 2)
    eq_(len(food_vocab_man), 12728)
    eq_(len(natr_vocab_man), 8972)
    eq_(sorted(food_vocab_man)[3200], 'ANALYSTS')
    eq_(sorted(natr_vocab_man)[3210], 'NGX')
    
    
def test_d1_5_es():
    global food_corpus, natr_corpus
    
    food_corpus_tk = train.tokenize_corpus(food_corpus)
    natr_corpus_tk = train.tokenize_corpus(natr_corpus)
    
    food_corpus_tk_pd = train.pad_corpus(food_corpus_tk)
    natr_corpus_tk_pd = train.pad_corpus(natr_corpus_tk)
    
    food_corpus_tr, food_corpus_te = train.split_corpus(food_corpus_tk_pd)
    natr_corpus_tr, natr_corpus_te = train.split_corpus(natr_corpus_tk_pd)
    
    food_ngrams, food_vocab_man = train.count_ngrams(food_corpus_tr, 3)
    natr_ngrams, natr_vocab_man = train.count_ngrams(natr_corpus_tr, 3)
    
    eq_(train.estimate(food_ngrams, ['palm'], ['producer', 'of']), 0.25)
    eq_(train.estimate(natr_ngrams, ['basis'], ['tested', 'the']), 0.5)


def test_d2_1_gp():
    global food_corpus, natr_corpus
    
    food_corpus_tk = train.tokenize_corpus(food_corpus)
    natr_corpus_tk = train.tokenize_corpus(natr_corpus)
    
    food_train, food_vocab = padded_everygram_pipeline(3, food_corpus_tk[:int(0.8*len(food_corpus_tk))])
    natr_train, natr_vocab = padded_everygram_pipeline(3, natr_corpus_tk[:int(0.8*len(natr_corpus_tk))])

    food_test = sum([['<s>'] + x + ['</s>'] for x in food_corpus_tk[int(0.8*len(food_corpus_tk)):]],[])
    natr_test = sum([['<s>'] + x + ['</s>'] for x in natr_corpus_tk[int(0.8*len(natr_corpus_tk)):]],[])

    food_lm = Laplace(3)
    natr_lm = Laplace(3)

    food_lm.fit(food_train, food_vocab)
    natr_lm.fit(natr_train, natr_vocab)
    
    eq_(int(evaluate.get_perplexity(food_lm, food_test[:2500])), 7318)
    eq_(int(evaluate.get_perplexity(food_lm, natr_test[:2500])), 7309)
    eq_(int(evaluate.get_perplexity(natr_lm, natr_test[:2500])), 5222)
    eq_(int(evaluate.get_perplexity(natr_lm, food_test[:2500])), 5354)

    
def test_d3_1_vary():
    global food_corpus, natr_corpus
    
    food_corpus_tk = train.tokenize_corpus(food_corpus)
    
    n_gram_orders = [2, 3]
    
    train_corpus = food_corpus_tk[:int(0.8*len(food_corpus_tk))]
    test_corpus = food_corpus_tk[int(0.8*len(food_corpus_tk)): int(0.85*len(food_corpus_tk))]

    results = train.vary_ngram(train_corpus, test_corpus, n_gram_orders)
    
    eq_(int(results[2]), 7387)
    eq_(int(results[3]), 7428)

