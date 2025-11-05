#!/usr/bin/env python3

import rospy
import queue
import numpy as np
from scipy.stats import norm

from rosneuro_msgs.msg import NeuroOutput, NeuroEvent
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty


class continous_hmm:
    def __init__(self):
        rospy.init_node('continous_hmm')
        self.setup_params()
        self.setup_subscribers()
        self.setup_publishers()
        self.setup_services()

    def setup_params(self):
        self.rate = rospy.get_param('~rate', 16)
        self.class_1 = 773 # BH   -> left
        self.class_2 = 771 # BF   -> right
        self.class_c = 783 # rest -> front
        self.has_new_classdata = False
        self.sample_size = rospy.get_param('~sample_size', 8)
        self.sample_queue = queue.Queue()

        self.trav_matrix = np.ones((3, 3))/9

        self.posterior_prob = np.ones(3)/3
        self.mean = np.array([0.9, 0.1])
        self.st   = np.array([0.1, 0.1])

    def reset(self):
        self.posterior_prob = np.ones(3)/3
        self.sample_queue = queue.Queue()

    def setup_services(self):
        self.serv_res = rospy.Service('/hmm/reset', Empty, self.reset)

    def setup_subscribers(self):
        self.sub_trav = rospy.Subscriber('/hmm/traversability_matrix', Float64MultiArray, self.callback_traversability)
        self.sub_smr  = rospy.Subscriber('/smrbci/neuroprediction', NeuroOutput, self.callback_smr)

    def setup_publishers(self):
        self.pub = rospy.Publisher('/hmm/neuroprediction', NeuroOutput, queue_size=10)

    def callback_traversability(self, msg : Float64MultiArray):
        self.trav_matrix = np.array(msg.data).reshape((3, 3))

    def callback_smr(self, msg : NeuroOutput):
        # Get the relevant class
        idx_class_1 = np.argwhere(np.array(msg.decoder.classes) == self.class_1)[0][0]

        # Since it is a binary classifier I always have p and (1-p)
        new_sample = msg.softpredict.data[idx_class_1]

        # I update the sequence of sample that I am keeping track
        self.update_queue_sample(new_sample)

        self.has_new_classdata = True

    def update_queue_sample(self, new_sample):
        if self.sample_queue.qsize() > self.sample_size:
            self.sample_queue.get()
        self.sample_queue.put(new_sample)

    def compute_likelihood(self):
        samples = np.array(self.sample_queue.queue)

        # init the likelihood
        p_class_1 = 1
        p_class_c = 1
        p_class_2 = 1

        # compute the likelihood
        for sample in samples:
            # For now only use the expected distribution
            p_class_1 *=  norm.pdf(sample, loc=self.mean[0], scale=self.st[0])
            p_class_c *= (norm.pdf(sample, loc=self.mean[0], scale=self.st[0]) * (0.5) + \
                          norm.pdf(sample, loc=self.mean[1], scale=self.st[1]) * (0.5))
            p_class_2 *=  norm.pdf(sample, loc=self.mean[1], scale=self.st[1])

        # normalize the likelihood
        n_sample = len(samples)
        p_class_1 = p_class_1 / n_sample
        p_class_c = p_class_c / n_sample
        p_class_2 = p_class_2 / n_sample

        likelihood = np.array([p_class_1, p_class_c, p_class_2])

        return likelihood


    def compute_step(self):
        prediction = np.matmul(self.posterior_prob, self.trav_matrix)
        likelihood = self.compute_likelihood()
        self.posterior_prob = prediction * likelihood
        self.posterior_prob = self.posterior_prob / np.sum(self.posterior_prob)

    def update_msg(self):
        self.msg_prediction = NeuroOutput()
        self.msg_prediction.softpredict.data = list(self.posterior_prob)
        self.msg_prediction.decoder.type = "Hmm"
        self.msg_prediction.decoder.classes = [self.class_1, self.class_c, self.class_2]

    def publish_msg(self):
        self.pub.publish(self.msg_prediction)

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.has_new_classdata:
                self.compute_step()
                self.has_new_classdata = False
            self.update_msg()
            self.publish_msg()

            rate.sleep()


def main():
    chmm = continous_hmm()
    chmm.run()

if __name__ == '__main__':
    main()
