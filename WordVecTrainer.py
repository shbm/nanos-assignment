from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from time import time
import logging

logging.basicConfig(format='%(asctime)s:%(msecs)d %(filename)s:%(lineno)s - %(funcName)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
log = logging.getLogger(__name__)


class WordVecTrainer:
    def __init__(self):
        self.sentences = None
        self.w2v_model = None

    def train(self, sentences, cores=1):
        self.sentences = sentences
        sent = [row.split() for row in sentences]
        phrases = Phrases(sent, min_count=1, progress_per=10000)
        bigram = Phraser(phrases)
        sentences = bigram[sent]
        log.info('Building Word Vocabulary Model')
        w2v_model = Word2Vec(size=50, min_count=1, window=3, alpha=0.01, min_alpha=0.0007, workers=cores)

        t = time()
        w2v_model.build_vocab(sentences, progress_per=10000)
        log.info('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
        t = time()
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=300, report_delay=1)
        log.info('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
        w2v_model.init_sims(replace=True)

        self.w2v_model = w2v_model

    def get_similar_words(self, word, top_n=5):
        if word in self.w2v_model.wv.vocab:
            s = self.w2v_model.wv.most_similar(positive=word, topn=top_n)
            return [i[0] for i in s]
        else:
            return -1