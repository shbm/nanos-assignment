from bs4 import BeautifulSoup
from tqdm import tqdm
from bs4.element import Comment
import re
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
import multiprocessing
from WordVecTrainer import WordVecTrainer
import logging
import click
import crayons

nltk.download('stopwords')

logging.basicConfig(format='%(asctime)s:%(msecs)d %(filename)s:%(lineno)s - %(funcName)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
log = logging.getLogger(__name__)
CHROME_DRIVER_LOCATION = ChromeDriverManager().install()


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def decontracted(phrase):
    """
    Decontractions of a phrase. For eg. won't -> will not
    :param phrase: str for phrase
    :return: str decontracted phrase
    """
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def get_text_from_html(page_source):
    """
    Extract list of sentences from html page source
    :param page_source: html content
    :return: list of all the sentences in the html content
    """
    soup = BeautifulSoup(page_source, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    s = [t.strip() for t in visible_texts]
    return s


def get_html_string(url):
    """
    Gets HTML page source of a URL
    :param url: url of a webpage
    :return: get the page source of the url
    """
    print(f'Downloading webpage: {url}')

    driver = webdriver.Chrome(CHROME_DRIVER_LOCATION)
    driver.get(url)
    time.sleep(5)

    page_source = driver.page_source
    driver.quit()
    return page_source


def get_corpus(url):
    """
    Get list of all sentences given a URL post preprocessing.
    The function returns words which are decontracted, lowercase and stopwords removed.
    :param url: URL of a webpage
    :return: list of all sentences from the webpage
    """
    html = get_html_string(url)
    lines = get_text_from_html(html)

    sentences = []
    log.info('Cleaning the webpage content.')
    for sentance in tqdm(lines):
        sentance = re.sub(r"http\S+", "", sentance)
        sentance = BeautifulSoup(sentance, 'lxml').get_text()
        sentance = decontracted(sentance)
        sentance = re.sub("\S*\d\S*", "", sentance).strip()
        sentance = re.sub('[^A-Za-z]+', ' ', sentance)
        sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords.words())
        sentences.append(sentance.strip())

    s = [i for i in sentences if len(i) > 1]
    return s


def get_word_frequency(sentences):
    """
    Returns a defaultdict of word frequency.
    :param senteces: list of list containing strings
    :return: words frequency which is a dictionary
    """
    word_freq = defaultdict(int)
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1
    log.info(('Number of major unique words in the webpage: ',len(word_freq)))
    return word_freq


@click.command()
@click.option('-u', '--url', help='URL of the webpage', prompt=True)
@click.option('-w', '--words', help='Individual word provided one by one', multiple=True)
@click.option('-l', '--log', default=1, help='Use 0 to disable logs and 1 to enable logs', prompt=True)
def cli(url, words, log):
    if log == 0:
        logging.disable(logging.INFO)

    s = get_corpus(url)
    w2v = WordVecTrainer()

    cores = multiprocessing.cpu_count()-1
    w2v.train(s, cores=cores-1)
    for word in words:
        similar_words = w2v.get_similar_words(word)
        if similar_words != -1:
            s_ = ", ".join(similar_words)
            print(word, ":", crayons.blue(s_))
        else:
            print(word, ":", crayons.red('NOT FOUND in the vocabulary. Please check the spelling'))


if __name__ == '__main__':
    cli.main(auto_envvar_prefix='TEST')