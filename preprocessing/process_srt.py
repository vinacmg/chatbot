from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import Text, FreqDist
import json
import glob
import copy
import numpy as np

try:
	from .srt_to_string import subs_to_list
except: 
	from srt_to_string import subs_to_list


def replace_all(string, proib_list, new_word=''):
	for word in proib_list:
		string = string.replace(word, new_word)

	return string


def build_dictionaries(tokens, vocab_size):

	num2word = {}
	word2num = {}

	fdist = FreqDist(tokens)
	most_common = fdist.most_common(vocab_size)
	sort = sorted(most_common)
	
	for i in range(3, vocab_size):
		num2word[i] = sort[i-3][0]
		word2num[sort[i-3][0]] = i

	num2word[0] = "PAD"
	num2word[1] = "EOS"
	num2word[2] = "UNK"
	word2num["PAD"] = 0
	word2num["EOS"] = 1
	word2num["UNK"] = 2

	return num2word, word2num

def save_dictionaries(num2word, word2num):
	try:
		with open('dictionaries/num2word.txt', 'w') as f:

			json.dump(num2word, f)

		with open('dictionaries/word2num.txt', 'w') as f:

			json.dump(word2num, f)
	except:
		with open('preprocessing/dictionaries/num2word.txt', 'w') as f:

			json.dump(num2word, f)

		with open('preprocessing/dictionaries/word2num.txt', 'w') as f:

			json.dump(word2num, f)

def load_dictionaries(directory):

	try:
		with open(directory + 'dictionaries/num2word.txt', 'r') as f:

			num2word = json.load(f)

		for i in range(0, len(num2word)):
			num2word[i] = num2word[str(i)]
			del num2word[str(i)]

		with open(directory + 'dictionaries/word2num.txt', 'r') as f:

			word2num = json.load(f)

		return num2word, word2num
	except:
		with open(directory + 'preprocessing/dictionaries/num2word.txt', 'r') as f:

			num2word = json.load(f)

		for i in range(0, len(num2word)):
			num2word[i] = num2word[str(i)]
			del num2word[str(i)]

		with open(directory + 'preprocessing/dictionaries/word2num.txt', 'r') as f:

			word2num = json.load(f)

		return num2word, word2num

def save_sentences(sentences_talked, sentences_answered, tokens):

	try:
		with open('sentences/sentences_talked.txt', 'w') as f:

			json.dump(sentences_talked, f)

		with open('sentences/sentences_answered.txt', 'w') as f:

			json.dump(sentences_answered, f)

		with open('sentences/tokens.txt', 'w') as f:

			json.dump(tokens, f)
	except:
		with open('preprocessing/sentences/sentences_talked.txt', 'w') as f:

			json.dump(sentences_talked, f)

		with open('preprocessing/sentences/sentences_answered.txt', 'w') as f:

			json.dump(sentences_answered, f)

		with open('preprocessing/sentences/tokens.txt', 'w') as f:

			json.dump(tokens, f)

def load_sentences(directory):

	try:
		with open(directory + 'sentences/sentences_talked.txt', 'r') as f:

			sentences_talked = json.load(f)

		with open(directory + 'sentences/sentences_answered.txt', 'r') as f:

			sentences_answered = json.load(f)

		with open(directory + 'sentences/tokens.txt', 'r') as f:

			tokens = json.load(f)

		return sentences_talked, sentences_answered, tokens

	except:
		with open(directory + 'preprocessing/sentences/sentences_talked.txt', 'r') as f:

			sentences_talked = json.load(f)

		with open(directory + 'preprocessing/sentences/sentences_answered.txt', 'r') as f:

			sentences_answered = json.load(f)

		with open(directory + 'preprocessing/sentences/tokens.txt', 'r') as f:

			tokens = json.load(f)

		return sentences_talked, sentences_answered, tokens

def save_ids(ids_talked, ids_answered):
	try:
		with open('ids/ids_talked.txt', 'w') as f:

			json.dump(ids_talked, f)

		with open('ids/ids_answered.txt', 'w') as f:

			json.dump(ids_answered, f)

	except:
		with open('preprocessing/ids/ids_talked.txt', 'w') as f:

			json.dump(ids_talked, f)

		with open('preprocessing/ids/ids_answered.txt', 'w') as f:

			json.dump(ids_answered, f)

def load_ids(directory):
	try:
		with open(directory + 'ids/ids_talked.txt', 'r') as f:

			ids_talked = json.load(f)

		with open(directory + 'ids/ids_answered.txt', 'r') as f:

			ids_answered = json.load(f)

		return ids_talked, ids_answered

	except:
		with open(directory + 'preprocessing/ids/ids_talked.txt', 'r') as f:

			ids_talked = json.load(f)

		with open(directory + 'preprocessing/ids/ids_answered.txt', 'r') as f:

			ids_answered = json.load(f)

		return ids_talked, ids_answered

###START####	

dir_list = glob.glob('../srt/*.srt')

tags = ['<i>','</i>','{i}','{/i}','<b>','</b>','{b}','{/b}','<u>','</u>','{u}','{/u}','\"','\''] #problemas com aspas

sentences_talked, sentences_answered, tokens = load_sentences('')

'''
def process_srt(directory):

	sublist = subs_to_list(directory)

	no_tags_sents = [replace_all(sent, tags) for sent in sublist]
	lower_sents = [sent.lower() for sent in no_tags_sents]

	sentences = [word_tokenize(sent) for sent in lower_sents]
	sentences = sentences[3:-3] #retirar possíveis créditos da legenda

	tokens = []

	for i in sentences:
		for j in i:
			tokens.append(j)

	return sentences, tokens


sentences_talked = []
sentences_answered = []
tokens = []

for srt in dir_list: #concat sentences of all subs

	sentences_all, tkns = process_srt(srt)

	sentences_talked += copy.deepcopy(sentences_all[:-1]) #copy values. not references
	sentences_answered += copy.deepcopy(sentences_all[1:])
	tokens += (tkns)

save_sentences(sentences_talked, sentences_answered, tokens)
'''
'''
##########CUILDADO
dict_size = 40000
num2word, word2num = build_dictionaries(tokens, dict_size)
save_dictionaries(num2word ,word2num)
#############
'''
num2word, word2num = load_dictionaries("")

#sentences of 'word ids'
#fazer função que faz isso e salva 
for sent in sentences_talked:
	for i in range(0, len(sent)):
		try:
			sent[i] = word2num[sent[i]]
		except:
			sent[i] = word2num['UNK']

for sent in sentences_answered:
	for i in range(0, len(sent)):
		try:
			sent[i] = word2num[sent[i]]
		except:
			sent[i] = word2num['UNK']
#
#sentences of 'word ids'

#save_ids(sentences_talked, sentences_answered)
ids_talked, ids_answered = load_ids("")

def build_matrices(max_length):

	talks_seq_length = []
	answers_seq_length = []
	y = []
	reverse = [sent[::-1] for sent in ids_talked]

	for sent in reverse:

		talks_seq_length.append(len(sent))

		if(len(sent) < max_length):
			sent += ([word2num["PAD"]]*(max_length - len(sent)))

		elif(len(sent) > max_length):
			print("ERROR: Set a max_length >= the max sentence lenght")
			return

	for sent in ids_answered:

		answers_seq_length.append(len(sent)+1)

		if(len(sent) < max_length):

			y.append([word2num["EOS"]] + copy.deepcopy(sent) + [word2num["PAD"]]*(max_length - len(sent) - 1))
			sent.append(word2num["EOS"])
			sent += ([word2num["PAD"]]*(max_length - len(sent)))

		else:
			print("ERROR: Set a max_length >= the max sentence (for asnwered senteces) lenght")
			return

	x = np.transpose(np.matrix(reverse))
	y = np.transpose(np.matrix(y))
	target = np.transpose(np.matrix(ids_answered))
	talks_seq_length = np.array(talks_seq_length)
	answers_seq_length = np.array(answers_seq_length)
	
	return x, talks_seq_length, y, target, answers_seq_length

def d_size():

	return len(sentences_talked)

#print(len(set(tokens)))
#print(len(sentences_talked))

print("process_srt executed...")

