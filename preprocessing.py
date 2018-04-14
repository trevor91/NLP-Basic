import re
import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from multiprocessing import Pool

class preprocessing(object):
	@staticmethod
	def review_to_wordlist(review, remove_stopwords = False):
		review_text = BeautifulSoup(review, 'html.parser').get_text()
		review_text = re.sub('[^a-zA-Z]',' ',review_text)
		words = review_text.lower().split()

		if remove_stopwords:
			stops = set(stopwords.words('english'))
			words = [w for w in words if not w in stops]

		stemmer = SnowballStemmer('english')
		words = [stemmer.stem(w) for w in words]
		return words

	@staticmethod
	def review_to_sentences(review, remove_stopwords = False):
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		raw_sentences = tokenizer.tokenize(review.strip())
		sentences = []
		for raw_sentence in raw_sentences:
			if len(raw_sentence) > 0:
				sentences.append( preprocessing.review_to_wordlist(raw_sentence, remove_stopwords))
		return sentences