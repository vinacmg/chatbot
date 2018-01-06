import numpy as np
import tensorflow as tf
from preprocessing.process_srt import build_matrices, d_size, load_sentences
from preprocessing.process_chat import ProcessChat



class Model:

	####Settings##########
	max_time = 20
	data_size = 6740
	input_vocab_size = 5333
	target_vocab_size = 5333 - 1 # -1 por causa do UNK
	embedding_size = 100 #for large data probably 200 is good
	hidden_units = 256 #set 1024 later
	num_layers = 2
	batch_size = 20
	data_sections = data_size/batch_size
	epochs = 7
	learning_rate = 0.015
	init_minval_lstm = -0.08
	init_maxval_lstm = 0.08
	######################

	vars_dict = {}

	encoder_input_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name='encoder_inputs')
	encoder_sequence_length = tf.placeholder(shape=[None], dtype=tf.int32, name='encoder_sequence_length')
	decoder_input_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name='encoder_inputs')
	decoder_sequence_length = tf.placeholder(shape=[None], dtype=tf.int32, name='decoder_sequence_length')
	decoder_targets_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name='encoder_inputs')

	embeddings = tf.Variable(tf.random_uniform([target_vocab_size, embedding_size], -0.8, 0.8), dtype=tf.float32)

	encoder_inputs = tf.nn.embedding_lookup(embeddings, encoder_input_ids) 
	decoder_inputs = tf.nn.embedding_lookup(embeddings, decoder_input_ids)

	#Encoder######################
	if num_layers > 1:
		stacked_encoder = []
		for layer in range(num_layers):
			stacked_encoder.append(tf.nn.rnn_cell.LSTMCell(num_units = hidden_units, state_is_tuple=True, initializer=tf.random_uniform_initializer(init_minval_lstm, init_maxval_lstm)))
		encoder_cell = tf.nn.rnn_cell.MultiRNNCell(stacked_encoder, state_is_tuple=True)
	else: encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units = hidden_units, state_is_tuple=True, initializer=tf.random_uniform_initializer(init_minval_lstm, init_maxval_lstm))

	encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
		cell=encoder_cell,
		dtype=tf.float32,
		sequence_length=encoder_sequence_length,
		inputs=encoder_inputs,
		time_major=True,
		scope="encoder")
	###############################

	#Decoder#######################
	#weights of output projection
	W = tf.Variable(tf.random_uniform([hidden_units, target_vocab_size], -0.8, 0.8), dtype=tf.float32)
	#biases of output projection
	b = tf.Variable(tf.zeros([target_vocab_size]), dtype=tf.float32)

	if num_layers > 1:
		stacked_decoder = []
		for layer in range(num_layers):
			stacked_decoder.append(tf.nn.rnn_cell.LSTMCell(num_units = hidden_units, state_is_tuple=True, initializer=tf.random_uniform_initializer(init_minval_lstm, init_maxval_lstm)))
		decoder_cell = tf.nn.rnn_cell.MultiRNNCell(stacked_decoder, state_is_tuple=True)
	else: decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units = hidden_units, state_is_tuple=True, initializer=tf.random_uniform_initializer(init_minval_lstm, init_maxval_lstm))


	decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
		cell=decoder_cell,
		dtype=tf.float32,
		sequence_length=decoder_sequence_length,
		initial_state=encoder_final_state,
		inputs=decoder_inputs,
		time_major=True,
		scope="decoder")
	###############################

	decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
	decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
	decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
	decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, target_vocab_size))

	stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
		labels=tf.one_hot(decoder_targets_ids, depth=target_vocab_size, dtype=tf.float32),
		logits=decoder_logits,)

	loss = tf.reduce_mean(stepwise_cross_entropy)
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	def __init__(self):

		self.x, self.enc_seq_length, self.y, self.target, self.dec_seq_length = build_matrices(self.max_time)

		self.prcss_chat = ProcessChat()

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder'):
			self.vars_dict[v.name] = v

	def train(self):

		x = self.x
		y = self.y
		target = self.target
		enc_seq_length = self.enc_seq_length
		dec_seq_length = self.dec_seq_length
		sess = self.sess

		##### batch 0########




		##### batch 1 and forward ######
		for epoch in range(0, epochs):
			for batch in range(1, data_sections):
				input_x = x[:,batch*batch_size:(batch+1)*batch_size]
				input_y = y[:,batch*batch_size:(batch+1)*batch_size]
				target_y = target[:,batch*batch_size:(batch+1)*batch_size]

				_, l = (sess.run([train_op, loss], feed_dict={encoder_input_ids: input_x, 
					decoder_input_ids: input_y, 
					encoder_sequence_length: enc_seq_length, 
					decoder_targets_ids: target_y,
					decoder_sequence_length: dec_seq_length}))


				#loss_track.append(l)


	def inference(self, chat_input):

		#put option: with feed_previous or without
		#optimize feed_previous (sem gambiarra)

		decoder_logits = self.decoder_logits

		input_x = np.transpose(np.matrix(chat_input))
		print(input_x)
		input_y = [1]
		matrix_y = np.transpose(np.matrix(input_y))
		enc_seq_length = np.array([len(input_x)])

		while(True):

			dec_seq_length = np.array([len(matrix_y)])

			decoder_probabilites = tf.nn.softmax(decoder_logits)
			decoder_prediction = tf.argmax(decoder_probabilites, 2) #[max-time,batch-size]

			prediction = sess.run(decoder_prediction, feed_dict={encoder_input_ids: input_x, 
					decoder_input_ids: matrix_y, 
					encoder_sequence_length: enc_seq_length,
					decoder_sequence_length: dec_seq_length})

			if(prediction[-1] == [1]):
				return prediction[:-1]

			matrix_y = np.concatenate((np.matrix([1]), prediction),)

'''
	def init_lstm_weights(self, minval, maxval): #use this logic if lstmcell initializer be deprecated or using other cell types

		vars_dict = self.vars_dict
		hidden_units = self.hidden_units
		embedding_size = self.embedding_size
		var1 = (tf.Variable(tf.random_uniform([(hidden_units+embedding_size), 4*hidden_units], minval, maxval), dtype=tf.float32))
		var2 = (tf.Variable(tf.random_uniform([(hidden_units*2), 4*hidden_units], minval, maxval), dtype=tf.float32))
		self.sess.run(var1.initializer)
		self.sess.run(var2.initializer)

		self.sess.run(vars_dict['decoder/multi_rnn_cell/cell_0/lstm_cell/kernel:0'].assign(
			var1
			))

		self.sess.run(vars_dict['decoder/multi_rnn_cell/cell_1/lstm_cell/kernel:0'].assign(
			var2
			))
'''

model = Model()

for v in model.vars_dict:
	print(v)

print(model.sess.run(model.vars_dict['decoder/multi_rnn_cell/cell_0/lstm_cell/kernel:0']))
print()
print(model.sess.run(model.vars_dict['decoder/multi_rnn_cell/cell_0/lstm_cell/bias:0']))
print()
print(model.sess.run(model.vars_dict['decoder/multi_rnn_cell/cell_1/lstm_cell/kernel:0']))
print()
print(model.sess.run(model.vars_dict['decoder/multi_rnn_cell/cell_1/lstm_cell/bias:0']))

'''
print(model.x[:,:10])
print(model.enc_seq_length[:10])
print(model.y[:,:10])
print(model.target[:,:10])
print(model.dec_seq_length[:10])
'''