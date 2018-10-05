import re
import logging
from gensim.models import Word2Vec
from bs4 import BeautifulSoup


def strip_tags(html):
	"""Strip html tags"""
	return BeautifulSoup(html).get_text(' ')

def text_to_token(text):
	"""Get list of token for training"""
	# Strip HTML
	text = strip_tags(text)
	# Keep only word
	text = re.sub("\W", " ", text)
	# Split sentence
	token = text.split()
	# Don't remember the number
	for i in range(len(token)):
		token[i] = '0' if token[i].isdigit() else token[i]
	return token

def read_sentences(filename):
	"""Read and split token from text file"""
	sentences = []
	with open(filename, 'r') as f:
		for line in f:
			if '|' not in line: # Remove menu items in some newspaper
				sentences.append(text_to_token(line.strip()))
	return sentences

if __name__ == '__main__':
	# Read data from files
	sentences = read_sentences("input.txt")
	'''print(sentences)
				print(len(sentences))'''
	print('Loaded {} sentences!'.format(len(sentences)))
	# Import the built-in logging module and configure it so that Word2Vec
	# creates nice output messages
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	# Set values for various parameters
	num_features = 300   # Word vector dimensionality: Số chiều vector
	min_word_count = 0    # Minimum word count: chưa rõ công dụng (Thử số lớn hơn thì bị thu giảm số từ, có khi lỗi)
	num_workers = 20       # Number of threads to run in parallel : Số luồng chạy song song
	print("Training Word2Vec model...")
	# Initialize and train the model
	model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count)
	model.init_sims(replace=True)

	#Save model
	'''ext_vec = "out_model.vec"
				model.save(ext_vec)'''
	ext_txt = 'all.w2v.txt'
	model.wv.save_word2vec_format(ext_txt, binary=False)
	'''ext_bin = 'out_model.bin'
				model.wv.save_word2vec_format(ext_bin, binary=True)'''