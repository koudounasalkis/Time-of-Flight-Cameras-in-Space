################################################################################################
#                                           TOF CLASS                                          #
################################################################################################

import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.interpolate

root_dir = '/Users/alkiskoudounas/Desktop/MasterThesis/Code/'
PI = 3.14159265358979323846
C = 0.000299792458
flg = False
dtype = tf.float32

class kinect_real_tf:

	def __init__(self):
		self.cam = kinect_real_tf_spec()		

		self.dtype= tf.float32
		self.rg = self.res_graph()
		self.dir = root_dir + '/params/kinect/'

		self.cam['delay'] = np.loadtxt(self.dir+'delay.txt',delimiter=',')
		self.cam['vig'] = np.loadtxt(self.dir+'vig.txt',delimiter=',')

		self.cam['raw_max'] = 3500 	# This brightness will be projected to 
		self.cam['map_max'] = 3500 	# This brightness will be the threshold for the kinect output
		self.cam['lut_max'] = 3800 	# This is the ending of lut table
		self.cam['sat'] = 32767 	# This is the saturated brightness

		self.cam['noise_samp'] = np.loadtxt(self.dir+'noise_samp_2000_notail.txt',delimiter=',')
		self.cam['val_lut'] = np.arange(self.cam['noise_samp'].shape[1])-\
			(self.cam['noise_samp'].shape[1]-1)/2

		self.gng = self.gain_noise_graph()
		self.gg = self.gain_graph()

		self.sess = tf.compat.v1.Session()
		self.sess.run(tf.compat.v1.global_variables_initializer())
		return


	def res_graph(self):
		cam = self.cam

		ipr_s = tf.compat.v1.placeholder(self.dtype, [None], name='ipr_s')

		ipr_idx = tf.compat.v1.placeholder(tf.int64, [3, None], name='ipr_idx')

		cha_num = 9
		cor = tf.compat.v1.placeholder(self.dtype, [cha_num,None], name='cor',)
		
		meas = []
		for i in range(cha_num):
			cor_cha = cor[i,:]
			cor_exp = tf.gather(cor_cha, ipr_idx[2,:])

			tmp = cor_exp * ipr_s
			tmp = tf.SparseTensor(tf.transpose(ipr_idx), tmp, [cam['dimy'],cam['dimx'],tf.reduce_max(ipr_idx[2,:])])
			tmp1 = tf.compat.v1.sparse_reduce_sum(tmp,2)
			meas.append(tmp1)

		meas = tf.stack(meas, 2)

		return {'meas':meas,'ipr_s':ipr_s,'ipr_idx':ipr_idx,'cor':cor,'tmp':tmp,'tmp1':tmp1}


	def res_delay_vig_graph(self):
		cam = self.cam

		ipr_s = tf.compat.v1.placeholder(self.dtype, [None], name='ipr_s')

		ipr_idx = tf.compat.v1.placeholder(tf.int64, [3, None], name='ipr_idx')

		delay_idx = tf.compat.v1.placeholder(self.dtype, [None], name='delay_idx')
		final_idx = tf.cast(ipr_idx[2,:],self.dtype)+delay_idx

		vig = tf.constant(self.cam['vig'],self.dtype)

		cha_num = 9
		cor = tf.compat.v1.placeholder(self.dtype, [cha_num,None], name='cor')
		
		meas = []
		for i in range(cha_num):
			cor_exp = tf.py_func(self.f[i],[final_idx], tf.float64)
			cor_exp = tf.cast(cor_exp, self.dtype)

			tmp = cor_exp * ipr_s
			tmp = tf.SparseTensor(tf.transpose(ipr_idx), tmp, [cam['dimy'],cam['dimx'],tf.reduce_max(ipr_idx[2,:])])
			tmp1 = tf.compat.v1.sparse_reduce_sum(tmp,2)
			meas.append(tmp1/vig)
 
		meas = tf.stack(meas, 2)

		return {\
			'meas':meas,
			'ipr_s':ipr_s,
			'ipr_idx':ipr_idx,
			'delay_idx':delay_idx,
			'cor':cor,
			'tmp':tmp,
			'tmp1':tmp1,
		}


	def res_delay_vig_motion_graph(self):
		cam = self.cam

		ipr_s = tf.compat.v1.placeholder(self.dtype, [9,None], name='ipr_s')

		ipr_idx = tf.compat.v1.placeholder(tf.int64, [3, None], name='ipr_idx')

		delay_idx = tf.compat.v1.placeholder(self.dtype, [9,None], name='delay_idx')

		vig = tf.constant(self.cam['vig'],self.dtype)

		cha_num = 9
		cor = tf.compat.v1.placeholder(self.dtype, [cha_num,None], name='cor',)
		
		meas = []
		for i in range(cha_num):
			final_idx = tf.cast(ipr_idx[2,:],self.dtype)+delay_idx[i,:]
			cor_exp = tf.py_func(self.f[i],[final_idx], tf.float64)
			cor_exp = tf.cast(cor_exp, self.dtype)

			tmp = cor_exp * ipr_s[i,:]
			tmp = tf.SparseTensor(tf.transpose(ipr_idx), tmp, [cam['dimy'],cam['dimx'],tf.reduce_max(ipr_idx[2,:])])
			tmp1 = tf.compat.v1.sparse_reduce_sum(tmp,2)
			meas.append(tmp1/vig)

		meas = tf.stack(meas, 2)

		return {\
			'meas':meas,
			'ipr_s':ipr_s,
			'ipr_idx':ipr_idx,
			'delay_idx':delay_idx,
			'cor':cor,
			'tmp':tmp,
			'tmp1':tmp1,
		}


	def gain_noise_graph(self):
		cam = self.cam

		raw_max = self.cam['raw_max']
		map_max = self.cam['map_max']
		lut_max = self.cam['lut_max']

		noise_samp = tf.constant(self.cam['noise_samp'], tf.int32)
		val_lut = tf.constant(self.cam['val_lut'], dtype=self.dtype)

		meas_i = tf.compat.v1.placeholder(self.dtype, [cam['dimy'],cam['dimx'],9], name='meas') 

		meas = meas_i * map_max / raw_max

		msk = tf.less(tf.abs(meas),lut_max)
		idx = tf.where(tf.abs(meas)<lut_max) 
		hf = tf.cast((tf.shape(noise_samp)[1]-1)/2,self.dtype)
		mean_idx = tf.cast(tf.boolean_mask(meas,msk)+hf, tf.int32)
		samp_idx = tf.cast(tf.compat.v1.random_uniform(\
			tf.shape(mean_idx),minval=0,maxval=self.cam['noise_samp'].shape[0],dtype=tf.int32\
		),tf.int32)
		idx_lut = tf.stack([samp_idx,mean_idx],1)
		idx_n = tf.gather_nd(noise_samp, idx_lut)
		noise = tf.gather(val_lut, idx_n, name='noise_samp')

		noise_s = tf.SparseTensor(idx, noise, tf.cast(tf.shape(meas),tf.int64))
		noise_s = tf.compat.v1.sparse_tensor_to_dense(noise_s)
		meas = tf.cast(noise_s, tf.int32)

		idx_thre = tf.where(tf.abs(meas)<map_max)
		flg = tf.ones(tf.shape(idx_thre[:,0]),tf.int32)
		flg_s = tf.SparseTensor(idx_thre, flg, tf.cast(tf.shape(meas),tf.int64))
		flg_s = tf.compat.v1.sparse_tensor_to_dense(flg_s)
		meas = meas * flg_s + (1-flg_s)*map_max

		meas_o = meas / map_max

		res_dict = {
			'meas_i'	:	meas_i, 
			'meas_o'	:	meas_o, 
			'noise'		:	noise,
			'mean_idx'	:	mean_idx,
			'idx_n'		:	idx_n,
			'idx_lut'	:	idx_lut,
			'noise_s'	:	noise_s,
		}

		return res_dict


	def gain_graph(self):
		cam = self.cam

		raw_max = self.cam['raw_max']
		map_max = self.cam['map_max']

		meas_i = tf.compat.v1.placeholder(\
			self.dtype, 
			[cam['dimy'],cam['dimx'],9],
			name='meas',
		) 

		meas = meas_i * map_max / raw_max

		idx_thre = tf.where(tf.abs(meas)<map_max)
		flg = tf.ones(tf.shape(idx_thre[:,0]),tf.int32)
		flg_s = tf.SparseTensor(idx_thre, flg, tf.cast(tf.shape(meas),tf.int64))
		flg_s = tf.compat.v1.sparse_tensor_to_dense(flg_s)
		
		meas = tf.cast(meas, tf.int32) * flg_s + (1-flg_s)*map_max
		meas_o = meas / map_max

		res_dict = {
			'meas_i'	:	meas_i, 
			'meas_o'	:	meas_o, 
		}

		return res_dict


	def process(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		max_len = int(2e6)
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})

		result = { 'meas' : meas }
		return result


	def process_no_noise(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		max_len = int(2e6) 
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		meas = self.sess.run(self.gg['meas_o'],feed_dict={self.gg['meas_i']:meas})

		result = { 'meas' : meas }
		return result


	def process_gt(self,cam,depth_true):
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		t = depth_true / (C/2)
		t_idx = t / self.cam['exp']
		t_idx[np.where(depth_true<1e-4)] = np.nan
		t_idx[np.where(t_idx>cor.shape[1])] = np.nan
		t_idx = scipy.misc.imresize(t_idx,(cam['dimy'],cam['dimx']),mode='F')

		self.f = []
		for i in range(cor.shape[0]):
		    self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		meas = [self.f[i](t_idx) for i in range(cor.shape[0])]
		meas = np.stack(meas, 2)
		meas /= self.cam['raw_max']
		meas[np.where(np.isnan(meas))] = 0

		result = { 'meas': meas }
		return result


	def process_gt_vig(self,cam,depth_true):
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		t = depth_true / (C/2)
		t_idx = t / self.cam['exp']
		t_idx[np.where(depth_true<1e-4)] = np.nan
		t_idx[np.where(t_idx>cor.shape[1])] = np.nan
		t_idx = scipy.misc.imresize(t_idx,(cam['dimy'],cam['dimx']),mode='F')

		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		meas = [self.f[i](t_idx)/self.cam['vig'] for i in range(cor.shape[0])]
		meas = np.stack(meas, 2)
		meas /= self.cam['raw_max']
		meas[np.where(np.isnan(meas))] = 0

		result = { 'meas': meas }
		return result


	def process_gt_vig_dist_surf(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		max_len = int(2e6)
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)
		
		vig = np.tile(np.expand_dims(self.cam['vig'],-1),[1,1,9])
		meas /= vig
		meas /= self.cam['raw_max']
		meas[np.where(np.isnan(meas))] = 0

		result = { 'meas': meas }
		return result


	def process_gt_vig_dist_surf_mapmax(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		max_len = int(2e6)
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)
		
		vig = np.tile(np.expand_dims(self.cam['vig'],-1),[1,1,9])
		meas /= vig
		meas /= self.cam['raw_max']
		meas[np.where(np.isnan(meas))] = 0

		result = {'meas': meas}
		return result	


	def process_one_bounce(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)
		cam = nonlinear_adjust(self.cam,cor)

		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		max_len = int(2e6) 
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		meas = self.sess.run(self.gg['meas_o'],feed_dict={self.gg['meas_i']:meas})

		result = { 'meas': meas }
		return result


	def process_one_bounce_noise(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)
		cam = nonlinear_adjust(self.cam,cor)

		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		max_len = int(2e6)
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rg['meas'],
				feed_dict={\
					self.rg['ipr_s']:ipr_s[i:end],\
					self.rg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rg['cor']:cor,\
				}
			)

		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})

		result = { 'meas': meas }
		return result


	def process_delay_vig_gain_noise(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		if not hasattr(self, 'rdvg'):
			self.rdvg = self.res_delay_vig_graph()

		delay_idx = self.cam['delay'][ipr_idx[0:2]]
		delay_idx /= (C/2)
		delay_idx /= self.cam['exp']

		max_len = int(1e7) 
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rdvg['meas'],
				feed_dict={\
					self.rdvg['ipr_s']:ipr_s[i:end],\
					self.rdvg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rdvg['delay_idx']:delay_idx[i:end],\
					self.rdvg['cor']:cor,\
				}
			)

		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})

		result = { 'meas'	: meas }
		return result


	def process_delay_vig_gain(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		if not hasattr(self, 'rdvg'):
			self.rdvg = self.res_delay_vig_graph()

		delay_idx = self.cam['delay'][ipr_idx[0:2]]
		delay_idx /= (C/2)
		delay_idx /= self.cam['exp']

		max_len = int(1e7) 
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rdvg['meas'],
				feed_dict={\
					self.rdvg['ipr_s']:ipr_s[i:end],\
					self.rdvg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rdvg['delay_idx']:delay_idx[i:end],\
					self.rdvg['cor']:cor,\
				}
			)

		meas = self.sess.run(self.gg['meas_o'],feed_dict={self.gg['meas_i']:meas})

		result = { 'meas' : meas }
		return result


	def process_gt_delay_vig_gain_noise(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)
		cam = nonlinear_adjust(self.cam,cor)

		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		if not hasattr(self, 'rdvg'):
			self.rdvg = self.res_delay_vig_graph()

		delay_idx = self.cam['delay'][ipr_idx[0:2]]
		delay_idx /= (C/2)
		delay_idx /= self.cam['exp']

		max_len = int(1e7)
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rdvg['meas'],
				feed_dict={\
					self.rdvg['ipr_s']:ipr_s[i:end],\
					self.rdvg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rdvg['delay_idx']:delay_idx[i:end],\
					self.rdvg['cor']:cor,\
				}
			)

		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})

		result = { 'meas': meas }
		return result


	def process_gt_delay_vig_dist_surf_mapmax(self,cam,ipr_idx,ipr_s,scenes,depth_true):
		self.cam['dimt'] = cam['dimt']
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)
		cam = nonlinear_adjust(self.cam,cor)

		y=ipr_idx[0]
		x=ipr_idx[1]
		idx = y*self.cam['dimx']+x
		idx_u, I = np.unique(idx, return_index=True)

		ipr_idx = (ipr_idx[0][(I,)], ipr_idx[1][(I,)], ipr_idx[2][(I,)])
		ipr_s = ipr_s[(I,)]

		self.f = []
		for i in range(cor.shape[0]):
			self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		if not hasattr(self, 'rdvg'):
			self.rdvg = self.res_delay_vig_graph()

		delay_idx = self.cam['delay'][ipr_idx[0:2]]
		delay_idx /= (C/2)
		delay_idx /= self.cam['exp']

		max_len = int(1e7) 
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))
		for i in range(0,len(ipr_s),max_len):
			end = min(len(ipr_s),i+max_len)
			meas += self.sess.run(\
				self.rdvg['meas'],
				feed_dict={\
					self.rdvg['ipr_s']:ipr_s[i:end],\
					self.rdvg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rdvg['delay_idx']:delay_idx[i:end],\
					self.rdvg['cor']:cor,\
				}
			)

		meas /= self.cam['raw_max']

		result = { 'meas': meas }
		return result


	def process_motion_delay_vig_gain(self,cam,ipr_idx,ipr_s,scenes,depth_true,motion_delay, max_t):
		self.cam['dimt'] = max(cam['dimt'], int(max_t))
		self.cam['exp'] = cam['exp']
		cor = compute_cor(self.cam)

		self.f = []
		for i in range(cor.shape[0]):self.f.append(scipy.interpolate.interp1d(np.arange(cor.shape[1]),cor[i,:]))

		if not hasattr(self, 'rdvgm'):self.rdvmg = self.res_delay_vig_motion_graph()

		delay_idx = self.cam['delay'][ipr_idx[0:2]]
		delay_idx /= (C/2)
		delay_idx /= self.cam['exp']

		delay_indices = []
		for i in range(9):delay_indices.append(delay_idx+motion_delay[i,:])
		delay_idx = np.stack(delay_indices,0)

		max_len = int(1e7) 
		meas = np.zeros((self.cam['dimy'],self.cam['dimx'],9))

		for i in range(0,ipr_s.shape[1],max_len):
			end = min(ipr_s.shape[1],i+max_len)
			meas += self.sess.run(\
				self.rdvmg['meas'],
				feed_dict={\
					self.rdvmg['ipr_s']:ipr_s[:,i:end],\
					self.rdvmg['ipr_idx']:np.array(ipr_idx)[:,i:end],\
					self.rdvmg['delay_idx']:delay_idx[:,i:end],\
					self.rdvmg['cor']:cor,\
				}
			)

		result = { 'meas': meas}
		return result


	def add_gain_noise(self, meas):
		meas = self.sess.run(self.gng['meas_o'],feed_dict={self.gng['meas_i']:meas})
		return meas


	def dist_to_depth(self, dist):
		cam = self.cam
		dimx = cam['dimx']
		dimy = cam['dimy']
		dimx = tf.convert_to_tensor(dimx, dtype=tf.float32)
		dimy = tf.convert_to_tensor(dimy, dtype=tf.float32)
		dist_x_shape = dist.shape.as_list()[1]
		dist_y_shape = dist.shape.as_list()[0]
		xx,yy = tf.meshgrid(list(range(dist_x_shape)),list(range(dist_y_shape)))
		xc = (dist_x_shape-1)/2
		xc = tf.convert_to_tensor(xc, dtype=tf.float32)
		yc = (dist_y_shape-1)/2
		yc = tf.convert_to_tensor(yc, dtype=tf.float32)
		xx = tf.cast(xx, dtype=tf.float32)
		yy = tf.cast(yy, dtype=tf.float32)
		coeff = tf.tan(cam['fov_x']/2/180*np.pi)
		coeff = tf.cast(coeff, dtype=tf.float32)
		dist_x_shape = tf.convert_to_tensor(dist_x_shape, dtype=tf.float32)
		dist_y_shape = tf.convert_to_tensor(dist_y_shape, dtype=tf.float32)

		xx = (xx - xc)/dist_x_shape*dimx/((dimx-1)/2) * coeff
		yy = (yy - yc)/dist_y_shape*dimy/((dimx-1)/2) * coeff
		z_multiplier = 1/tf.sqrt(xx**2+yy**2+1)
		z_multiplier = tf.expand_dims(z_multiplier, axis=-1)
		depth = dist * z_multiplier
		return depth


	def depth_to_dist(self, depth, batch_size, flg):
		cam = self.cam
		dimx = tf.convert_to_tensor(cam['dimx'], dtype=tf.float32)
		dimy = tf.convert_to_tensor(cam['dimy'], dtype=tf.float32)
		dist_x_shape = depth.shape.as_list()[2]
		dist_y_shape = depth.shape.as_list()[1]
		xx, yy = tf.meshgrid(list(range(dist_x_shape)), list(range(dist_y_shape)))
		xx = tf.cast(xx, dtype=tf.float32)
		yy = tf.cast(yy, dtype=tf.float32)
		xc = tf.convert_to_tensor((dist_x_shape - 1) / 2, dtype=tf.float32)
		yc = tf.convert_to_tensor((dist_y_shape - 1) / 2, dtype=tf.float32)
		coeff = tf.tan(cam['fov_x'] / 2 / 180 * np.pi)
		coeff = tf.cast(coeff, dtype=tf.float32)

		dist_x_shape = tf.convert_to_tensor(dist_x_shape, dtype=tf.float32)
		dist_y_shape = tf.convert_to_tensor(dist_y_shape, dtype=tf.float32)

		xx = (xx - xc) / dist_x_shape * dimx / ((dimx - 1) / 2) * coeff
		yy = (yy - yc) / dist_y_shape * dimy / ((dimy - 1) / 2) * coeff

		z_multiplier = tf.sqrt(xx ** 2 + yy ** 2 + 1)
		if flg == True: 
			z_multiplier = z_multiplier
		else:           
			z_multiplier = 1 / z_multiplier
		z_multiplier = tf.expand_dims(tf.expand_dims(z_multiplier, axis=-1), axis=0)
		z_multiplier = tf.tile(z_multiplier, multiples=[batch_size, 1, 1, 1])
		dist_or_depth = depth * z_multiplier
		return dist_or_depth, z_multiplier


#######################################################
def kinect_real_tf_spec():
	cam = {}
	cam['dimx'] = 512
	cam['dimy'] = 424
	cam['fov_x'] = 70

	prms = np.loadtxt(root_dir + '/params/kinect/cam_func_params.txt',delimiter=',')

	coef = 4*PI/3e-4
	cam['T'] 		= np.array([coef/prms[0], coef/prms[3], coef/prms[7]])
	cam['phase'] 	= -np.array([\
		[prms[1]*PI, (prms[1]+2/3)*PI, (prms[1]-2/3)*PI],
		[prms[4]*PI, (prms[4]+2/3)*PI, (prms[4]-2/3)*PI],
		[prms[8]*PI, (prms[8]+2/3)*PI, (prms[8]-2/3)*PI],
	])
	cam['A'] = np.array([prms[2], prms[6], prms[9]])
	cam['m'] = prms[5]
	return cam


#######################################################
def compute_cor(cam):
	cam['dimt'] += 20
	cam['tabs']	= np.array([\
		cam['dimt']
		for i in range(len(cam['T']))
	])
	cam['t'] = (np.arange(cam['dimt']))*cam['exp']

	cor = [\
		cam['A'][i]*np.sin(2*PI/cam['T'][i]*cam['t']-cam['phase'][i,j])
		for i in range(len(cam['T'])) for j in range(len(cam['phase'][i,:]))
	]
	for i in range(3,6):
		cor[i] = np.maximum(np.minimum(cor[i],np.abs(cam['m'])),-np.abs(cam['m']))
	return np.array(cor)


#######################################################
def nonlinear_adjust(cam, cor):
	cor = np.reshape(np.transpose(cor),[-1,3,3])
	phase = cam['phase']
	T = cam['T']
	depth_gt = cam['t']*C/2

	tmp_Qs = [[cor[:,k,j] * np.sin(phase[k][j]) for j in range(phase.shape[1])] for k in range(phase.shape[0])]
	tmp_Is = [[cor[:,k,j] * np.cos(phase[k][j]) for j in range(phase.shape[1])] for k in range(phase.shape[0])]
	tmp_Q = np.stack([np.sum(np.stack(tmp_Qs[k],-1),-1) for k in range(phase.shape[0])],-1)
	tmp_I = np.stack([np.sum(np.stack(tmp_Is[k],-1),-1) for k in range(phase.shape[0])],-1)
	phase_pred = np.arctan2(tmp_Q, tmp_I)
	depth_pred = np.stack([phase_pred[:,k]*T[k]/2/PI * C/2 for k in range(3)],-1)

	depth_cri = depth_pred[:,1]
	depth_unwrap = []
	for k in range(3):
		unwrap = np.floor((depth_cri-depth_pred[:,k])/(T[k]*C/2)+0.5)
		depth_unwrap.append(depth_pred[:,k] + unwrap*(T[k]*C/2))
	depth_pred = np.stack(depth_unwrap,-1)

	cam['depth_atan'] = depth_pred
	cam['depth_true'] = depth_gt
	return cam
