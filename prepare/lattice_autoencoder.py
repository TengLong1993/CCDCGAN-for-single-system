import os
import sys
import pickle

import numpy as np
import tensorflow as tf

import prepare.data_transformation as dt

###################################################################function
#####activation
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

#####round
def threshold(x, val=0.5):
    x = tf.clip_by_value(x,0.5,0.5001) - 0.5
    x = tf.minimum(x * 10000,1) 
    return x

#####neuron networks
def decoder(z, batch_size=1, phase_train=True, reuse=False):

	strides = [1,2,2,2,1]
	with tf.variable_scope("gen",reuse=reuse):
		z = tf.reshape(z,(batch_size,1,1,1,z_size))
		g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size,4,4,4,64), strides=[1,1,1,1,1], padding="VALID")
		g_1 = lrelu(g_1)

		g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size,8,8,8,64), strides=strides, padding="SAME")
		g_2 = lrelu(g_2)

		g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size,16,16,16,64), strides=strides, padding="SAME")
		g_3 = lrelu(g_3)

		g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size,32,32,32,1), strides=[1,2,2,2,1], padding="SAME")
		g_4 = tf.nn.sigmoid(g_4)

		return g_4

def encoder(inputs, phase_train=True, reuse=False):
	leak_value = 0.2
	strides = [1,2,2,2,1]
	with tf.variable_scope("enc",reuse=reuse):
		d_1 = tf.nn.conv3d(inputs, weights['wae1'], strides=strides, padding="SAME")
		d_1 = lrelu(d_1, leak_value)

		d_2 = tf.nn.conv3d(d_1, weights['wae2'], strides=strides, padding="SAME") 
		d_2 = lrelu(d_2, leak_value)
        
		d_3 = tf.nn.conv3d(d_2, weights['wae3'], strides=strides, padding="SAME")  
		d_3 = lrelu(d_3, leak_value) 

		d_4 = tf.nn.conv3d(d_3, weights['wae4'], strides=[1,1,1,1,1], padding="VALID")     
		d_4 = tf.nn.tanh(d_4)

		return d_4
#####weight
weights = {}
def initialiseWeights():
	global weights
	xavier_init = tf.contrib.layers.xavier_initializer()

	weights['wg1'] = tf.get_variable("wg1", shape=[4, 4, 4, 64, z_size], initializer=xavier_init)
	weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
	weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
	weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 1, 64], initializer=xavier_init)

	weights['wae1'] = tf.get_variable("wae1", shape=[4, 4, 4, 1, 64], initializer=xavier_init)
	weights['wae2'] = tf.get_variable("wae2", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
	weights['wae3'] = tf.get_variable("wae3", shape=[4, 4, 4, 64, 64], initializer=xavier_init)
	weights['wae4'] = tf.get_variable("wae4", shape=[4, 4, 4, 64, z_size], initializer=xavier_init)    

	return weights

###########################################################################training
#####parameters
batch_size = 1
z_size     = 25
reg_l2     = 0.0e-6
ae_lr      = 0.0003
n_ae_epochs= 201
def lattice_autocoder(lattice_graph_path='./test_lattice/',encoded_graph_path='./test_encoded_lattice/',model_path='./test_model/'):#e.g.: la.lattice_autocoder(lattice_graph_path='./original_lattice_graph/',encoded_graph_path='./original_encoded_lattice/',model_path='./model/')
	tf.reset_default_graph()
	if not os.path.exists(encoded_graph_path):
		os.makedirs(encoded_graph_path)
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	#####train_function
	weights = initialiseWeights()
	x_vector = tf.placeholder(shape=[batch_size,32,32,32,1],dtype=tf.float32)
	z_vector = tf.placeholder(shape=[batch_size,1,1,1,z_size],dtype=tf.float32) 

	# Weights for autoencoder pretraining
	with tf.variable_scope('encoders') as scope1:
		encoded = encoder(x_vector, phase_train=True, reuse=False)
		scope1.reuse_variables()
		encoded2 = encoder(x_vector, phase_train=False, reuse=True)

	with tf.variable_scope('gen_from_dec') as scope2:
		decoded = decoder(encoded, phase_train=True, reuse=False)
		scope2.reuse_variables()
		decoded_test = decoder(encoded2,phase_train=False, reuse=True)

	# Round decoder output
	decoded = threshold(decoded)
	decoded_test = threshold(decoded_test)
	# Compute MSE Loss and L2 Loss
	mse_loss = tf.reduce_mean(tf.pow(x_vector - decoded, 2))
	mse_loss2 = tf.reduce_mean(tf.pow(x_vector - decoded_test, 2))
	para_ae = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wae','wg'])]
	l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in para_ae])
	ae_loss = mse_loss + reg_l2 * l2_loss
	optimizer_ae = tf.train.AdamOptimizer(learning_rate=ae_lr, name="Adam_AE").minimize(ae_loss,var_list=para_ae)

	saver = tf.train.Saver() 

	with tf.Session() as sess:  
		sess.run(tf.global_variables_initializer())    
		test_size,test_name_list,train_name_list=dt.train_test_split(path=lattice_graph_path,split_ratio=0.1)
		min_mse_test=1

		for epoch in range(n_ae_epochs):
			batch_name_list=dt.get_batch_name_list(train_name_list,batch_size=311)
			mse_tr = 0; mse_test = 0;
			for interation in range(len(batch_name_list)):
				inputs_batch=np.load(lattice_graph_path+batch_name_list[interation]+'.npy').reshape(batch_size,32,32,32,1)
				mse_l, _ = sess.run([mse_loss, optimizer_ae],feed_dict={x_vector:inputs_batch})#,o_vector:object_batch})
				mse_tr += mse_l
			
			test_batch_name_list=dt.get_batch_name_list(test_name_list,batch_size=1000)
			for interation in range(len(test_batch_name_list)):
				test_inputs_batch=np.load(lattice_graph_path+test_batch_name_list[interation]+'.npy').reshape(batch_size,32,32,32,1)
				mse_t = sess.run(mse_loss2,feed_dict={x_vector:test_inputs_batch})#,o_vector:test_object_batch})
				mse_test += mse_t
			print (epoch,' ',mse_tr/len(batch_name_list),' ',mse_test/len(test_batch_name_list))

			if min_mse_test > mse_test/len(test_batch_name_list) and mse_test/len(test_batch_name_list)<5e-5:
				min_mse_test=mse_test/len(test_batch_name_list)
				saver.save(sess, save_path = model_path + 'lattice.ckpt')
				total_name_list=test_name_list+train_name_list
				for name in total_name_list:
					savefilename=encoded_graph_path+name+'.npy'
					encoded_lattice=np.zeros(200)
					encoded_lattice[0:25]=encoded2.eval(feed_dict={x_vector:np.load(lattice_graph_path+name+'.npy').reshape(batch_size,32,32,32,1)}).reshape(25)
					np.save(savefilename,encoded_lattice)

###########################################################################training

def lattice_restorer(generated_2d_path='./generated_2d_graph/',genenrated_decoded_path='./generated_decoded_lattice/',model_path='./test_model/'):
	tf.reset_default_graph()
	if not os.path.exists(genenrated_decoded_path):
		os.makedirs(genenrated_decoded_path)
	#####train_function
	weights = initialiseWeights()
	x_vector = tf.placeholder(shape=[batch_size,32,32,32,1],dtype=tf.float32)
	z_vector = tf.placeholder(shape=[batch_size,1,1,1,z_size],dtype=tf.float32) 

	# Weights for autoencoder pretraining
	with tf.variable_scope('encoders') as scope1:
		encoded = encoder(x_vector, phase_train=True, reuse=False)
		scope1.reuse_variables()
		encoded2 = encoder(x_vector, phase_train=False, reuse=True)

	with tf.variable_scope('gen_from_dec') as scope2:
		decoded = decoder(encoded, phase_train=True, reuse=False)
		scope2.reuse_variables()
		decoded_test = decoder(encoded2,phase_train=False, reuse=True)

	# Round decoder output
	decoded = threshold(decoded)
	decoded_test = threshold(decoded_test)
	# Compute MSE Loss and L2 Loss
	mse_loss = tf.reduce_mean(tf.pow(x_vector - decoded, 2))
	mse_loss2 = tf.reduce_mean(tf.pow(x_vector - decoded_test, 2))
	para_ae = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wae','wg'])]
	l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in para_ae])
	ae_loss = mse_loss + reg_l2 * l2_loss
	optimizer_ae = tf.train.AdamOptimizer(learning_rate=ae_lr, name="Adam_AE").minimize(ae_loss,var_list=para_ae)

	restore_saver = tf.train.Saver() 

	with tf.Session() as sess:  
		sess.run(tf.global_variables_initializer())    
		test_size,test_name_list,train_name_list=dt.train_test_split(path=generated_2d_path,split_ratio=0.1)

		restore_saver.restore(sess,model_path+'lattice.ckpt')

		total_name_list=test_name_list+train_name_list
		for name in total_name_list:
			savefilename=genenrated_decoded_path+name+'.npy'
			ge=np.load(generated_2d_path+name+'.npy')
			ge=ge[0,0:25].reshape(batch_size,1,1,1,z_size)
			decoded_lattice=decoded_test.eval(feed_dict={encoded2:ge}).reshape(32,32,32)
			np.save(savefilename,decoded_lattice)
		
