from nltk.tokenize import word_tokenize, sent_tokenize
try:
	from process_srt import load_dictionaries
except:
	from .process_srt import load_dictionaries
import numpy as np
import json

class ProcessChat:

	def __init__(self):

		self.num2word, self.word2num = load_dictionaries("")
		print('new instance of ProcessChat...')

		return

	def process_in(self, inp):

		word2num = self.word2num
		
		inp_tokenized = word_tokenize(inp)
		inp_ids = []
		lower_words = [word.lower() for word in inp_tokenized]
		
		for word in lower_words:
			try:
				inp_ids.append(word2num[word])
			except:
				inp_ids.append(word2num['UNK'])

		return(inp_ids)

	def process_out(self, out):
		
		num2word = self.num2word
		sent = []

		for number in out:
			sent.append(num2word[number])

		return sent
