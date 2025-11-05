#!/usr/bin/python3

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import LaserScan

class compute_traversability:
    def __init__(self):
        rospy.init_node('compute_traversability')
        self.setup_params()
        self.setup_subscribers()
        self.setup_publishers()

        self.trav_matrix = np.ones((self.num_sectors, self.num_sectors)) * (1.0 / self.num_sectors)

        # I may wanna add a service that reset the traversabiliy, need to check this

        self.recived_msg = False

    def setup_params(self):
        self.rate         = rospy.get_param('~rate', 16)
        self.num_sectors  = rospy.get_param('~num_sectors', 3)
        self.simple_mode  = rospy.get_param('~simple_mode', True)
        self.min_distance = rospy.get_param('~min_distance', 4)
        self.help_on      = rospy.get_param('~help', True)
        # Simple-mode will be the only programmed for now: it only check if he need to compute low help or high help
        # The alternative is to create a matrix depending to the position of the obstacles near the robot

    def setup_subscribers(self):
        self.sub_laser = rospy.Subscriber('/fused_scan', LaserScan, self.callback_laser)

    def setup_publishers(self):
        self.pub_traversability = rospy.Publisher('/hmm/traversability_matrix', Float64MultiArray, queue_size=10)

    def callback_laser(self, msg : LaserScan):
        self.msg = msg
        self.recived_msg = True

    def send_message(self):
        msg = Float64MultiArray()
        msg.data = self.trav_matrix.flatten()
        self.pub_traversability.publish(msg)
        # TODO: add the information of the shape of the matrix

    def convert_to_traversability(self):
        if (not self.simple_mode):
            rospy.warn("Only simple mode is programmed, fallback in that configuration")

        if (not self.help_on):
            # The T matrix need to remain constant if to the normal probability if the help is disabled
            return

        # Only simple-mode is developed, so I will only look at these coordinates:
        field_of_view = np.array([[-np.pi/2-0.34,-np.pi/2+0.34],[0-0.34,0+0.34],[np.pi/2-0.34,np.pi/2+0.34]])

        directio_obj = np.array([False, False, False])
        direction_d  = np.array([np.inf, np.inf, np.inf])

        index_direction = 0
        index_laser = 0

        current_angle = self.msg.angle_min
        inc = self.msg.angle_increment
        laser_reading = np.array(self.msg.ranges)

        # for each sector determine the nearest object
        while (index_laser < laser_reading.size and index_direction < direction_d.size):
            # Check if you are inside the requested field of view
            if current_angle > field_of_view[index_direction][0]:
                if current_angle < field_of_view[index_direction][1]:
                    # Update if the object is the nearest
                    if laser_reading[index_laser] < direction_d[index_direction]:
                        direction_d[index_direction] = laser_reading[index_laser]
                else:
                    index_direction = index_direction + 1
            # Go to the next delta angle
            current_angle = current_angle + inc
            index_laser = index_laser + 1

        # for each sector determine if the nearest object is enough close
        for i in range(len(direction_d)):
            if direction_d[i] < self.min_distance:
                directio_obj[i] = True

        # reverse the order of the array in order to preserve the consistency
        directio_obj = directio_obj[::-1]
        # [ right, front, left ] -> [left, front, right]

        # Now I now if the distances are there or not I will create the matrix according
        n_obstacles = np.sum(directio_obj)

        # Reset of the matrix
        self.trav_matrix = np.ones((self.num_sectors, self.num_sectors)) * (1.0 / self.num_sectors)

        # rospy.loginfo("Number of obstacles: " + str(n_obstacles))

        if (n_obstacles == 1):
           self.create_matrix_1(directio_obj)
        elif (n_obstacles == 2):
           self.create_matrix_2(directio_obj)

    def create_matrix_2(self, directio_obj):
        i = np.argwhere(directio_obj == False)[0]

        l = np.array([0.1, 0.1, 0.1])
        l[i] = 1.5

        l = l / np.sum(l)

        self.trav_matrix = np.array([l, l, l])

    def create_matrix_1(self, directio_obj):
        i = np.argwhere(directio_obj == True)[0]

        l = np.array([1.1, 1.1, 1.1])
        l[i] = 0.01

        l = l / np.sum(l)

        self.trav_matrix = np.array([l, l, l])

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.recived_msg:
                self.recived_msg = False
                self.convert_to_traversability()
            self.send_message()
            rate.sleep()

def main():
    ct = compute_traversability()
    ct.run()

if __name__ == "__main__":
    main()
