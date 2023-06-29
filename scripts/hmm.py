#!/usr/bin/python3

import rospy
import queue
import numpy as np
from scipy.stats import norm
from rosneuro_msgs.msg import NeuroOutput, NeuroEvent

from dynamic_reconfigure.server import Server
from rosneuro_hmm.cfg import HmmConfig

def configure_callback(config, level):
    global bias_bf, bias_bh, bias_rest
    global mu, sd
    
    bias_bf = config['bias_bf']
    bias_bh = config['bias_bh']
    bias_rest = config['bias_rest']

    mu_bh = config['mu_bh']
    mu_bf = config['mu_bf']
    
    sigma = config['sigma']

    mu = np.array([mu_bh, 0.5, mu_bf])
    sd    = np.array([sigma, sigma, sigma]) 

    return config

def compute_likelihood(means,st,samples):
    global bias_bf, bias_bh, bias_rest
    
    bh_p   = 1 
    rest_p = 1 
    bf_p   = 1

    for sample in samples:
        #bh_p    *= norm.pdf(sample, loc=means[0], scale=st[0])
        bh_p  *= ((norm.pdf(sample, loc=means[0], scale=st[0])*(1 - bias_bh) + norm.pdf(sample, loc=means[2], scale=st[2])*(bias_bh)))
        rest_p  *= ((norm.pdf(sample, loc=means[0], scale=st[0])*(bias_rest) + norm.pdf(sample, loc=means[2], scale=st[2])*(1-bias_rest)))
        #bf_p    *= norm.pdf(sample, loc=means[2], scale=st[2])
        bf_p  *= ((norm.pdf(sample, loc=means[0], scale=st[0])*(bias_bf) + norm.pdf(sample, loc=means[2], scale=st[2])*(1-bias_bf)))
    
    bh_p   /=len(samples)
    rest_p /=len(samples)
    bf_p   /=len(samples)
    
    likelihood = np.stack([bh_p, rest_p, bf_p])

    return likelihood

def one_step_update(T, posterior_tm1, sample_t, means, sd, window = 1):
    """ 
    Function for computing the one step update for the HMM
    """
    global fifo
    f = np.array(fifo.queue)
    if (len(f) >= window ):
        fifo.get()
    fifo.put(sample_t)
    f = np.array(fifo.queue)
    
    prediction = np.matmul(posterior_tm1,T)
    likelihood   = compute_likelihood(means,sd,f)
    posterior  = prediction * likelihood
    posterior  = posterior / np.sum(posterior)
    
    return prediction, likelihood, posterior
    
def on_receive_data(msg: NeuroOutput):
    global predictive_prob, likelihoods, posterior_prob
    global mu, sd, probability_update_matrix, initial_probability
    global window

    predictive_prob, likelihoods, posterior_prob = one_step_update(probability_update_matrix,posterior_prob,  msg.softpredict.data[0], mu, sd,window)

def main():
    rospy.init_node('my_integrator')
    rospy.loginfo("HI")
    global fifo
    fifo = queue.Queue()
    
    global window
    window = 8
     
    global bias_bf, bias_bh, bias_rest
    srv = Server(HmmConfig,configure_callback)
   
    bias_bf   = rospy.get_param('~bias_bf'   ,0.0)
    bias_bh   = rospy.get_param('~bias_bh'   ,0.0)
    bias_rest = rospy.get_param('~bias_rest' ,0.5)
    

    global predictive_prob, likelihoods, posterior_prob
    predictive_prob = []
    likelihoods         = []
    posterior_prob  = []
    
    global mu, sd, probability_update_matrix, initial_probability
    mu_bf =     rospy.get_param('~mu_bf', 0.1)
    mu_bh = 1 - rospy.get_param('~mu_bx', 0.1)
    sigma =     rospy.get_param('~sigma', 0.1)

    mu = np.array([mu_bh, 0.5, mu_bf])
    sd    = np.array([sigma, sigma, sigma])

    # update Probability Matrix -> I assume to be unifrom
    probability_update_matrix = np.array([[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,1/3,1/3]])
    
    initial_probability = np.array([1/3,1/3,1/3])
    
    predictive_prob = initial_probability
    posterior_prob = initial_probability
    likelihoods = initial_probability

    rospy.Subscriber("/smrbci/neuroprediction", NeuroOutput, on_receive_data)
    
    pub = rospy.Publisher('/my_integrator/neuroprediction', NeuroOutput, queue_size=1)
        
    q = queue.Queue()
    mean_window = 1
    out = []
     
    Hz = 16
    rate = rospy.Rate(Hz)
    while not rospy.is_shutdown():
      msg = NeuroOutput()
      state = np.argmax(posterior_prob)
      f = np.array(q.queue)
      if (len(f) >= mean_window ):
             q.get()
      q.put(state)
      f = np.array(q.queue)
      state = np.mean(f)
      
      state = 1 - state/2
      msg.softpredict.data = np.array([state, 1-state])
      pub.publish(msg)

      rate.sleep()
    
if __name__ == "__main__":
    main()
