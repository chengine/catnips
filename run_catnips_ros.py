#%% 
import numpy as np
import torch
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R

#Import utilies
from catnips.purr_utils import *

# Ros imports
import rospy
from typing import Tuple
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import QuaternionStamped
import os 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CATNIPS_ROS:
    def __init__(self, sub_topic):
        # Input Params
        traj_name = rospy.get_param("catnips/traj_name")
        self.dt = 0.2
        # hold = rospy.get_param("catnips/hold")
        # laps = rospy.get_param("catnips/laps")

        self.catnips_init(0)

        # Publishers
        self.pos_pub = rospy.Publisher("gcs/setpoint/position",PointStamped,queue_size=1)
        self.att_pub = rospy.Publisher("gcs/setpoint/attitude",QuaternionStamped,queue_size=1)

        # Subscribers
        self.pose_sub = rospy.Subscriber(sub_topic, PointStamped, self.update_current_pos)

    def catnips_init(self, ind):
        # Ind: index of which trajectory start and end to use

        # This is from world coordinates to scene coordinates
        rot = R.from_euler('xyz', [0, 0., 30], degrees=True)
        world_transform = np.eye(4)
        world_transform[:3, :3] = rot.as_matrix()
        world_transform[:3, -1] = np.array([5., 0., 0.75])
        world_transform = np.linalg.inv(world_transform)

        # In the world frame
        x0 = np.array([4., 0.5, 1.])
        xf = np.array([6.5, 0.5, 1.])

        if world_transform is not None:
            x0 = world_transform[:3, :3] @ x0 + world_transform[:3, -1]
            xf = world_transform[:3, :3] @ xf + world_transform[:3, -1]

        # Grid in nerfstudio frame
        grid = np.array([
            [-3., 3.],
            [-3., 3.],
            [-1., 3.]
            ])   

        #Create robot body
        agent_lims = .24*np.array([[-1, 1], [-1, 1], [-0.3, 0.3]])

        # #Configs
        sigma = 0.01
        spline_deg = 8
        discretization = 100

        catnips_configs = {
            'spline_deg': spline_deg,
            'sigma': sigma,
            'discretization': discretization,
            'dilation': None
        }

        position_configs = {
            'start': x0,
            'end': xf
        }

        save_purr_fp = f'purr_data/purr_sigma_{sigma}'

        self.catnips_planner = CATNIPS(data_path=save_purr_fp, grid=grid, agent_body=agent_lims, 
                                configs=catnips_configs, position_configs=position_configs, 
                                get_density=None)

        # Execute

        self.derivs = {
            'vel0': np.zeros(3),
            'accel0': np.zeros(3),
            'jerk0': np.zeros(3)
        }

        t = np.linspace(0, 2*np.pi, 10, endpoint=False)
        start = np.stack([2.2*np.cos(t) + 5.3, 2.2*np.sin(t), 1.25*np.ones_like(t)], axis=-1)
        self.x0 = start[ind]    # This is in the world frame

        # This is now in the nerfstudio frame
        if world_transform is not None:
            start = (world_transform[:3, :3] @ start.T).T + world_transform[:3, -1][None, :]

        end = np.roll(start, 5, axis=0)

        self.xf = end[ind]  # in nerfstudio frame

        self.world_transform = world_transform
        self.r = 0.25

        print(f'Going to end state: {np.linalg.inv(self.world_transform[:3, :3]) @ (self.xf - self.world_transform[:3, -1])}')

    def catnips_replan(self, event=None):

        # Convert the current position into nerfstudio frame
        start = self.world_transform[:3, :3] @ self.x0 + self.world_transform[:3, -1]

        traj, success = self.catnips_planner.get_traj(start, xf=self.xf, N=20, derivs=self.derivs)

        pts_in_world = (np.linalg.inv(self.world_transform[:3, :3]) @ (traj[:, :3] - self.world_transform[:3, -1][None, :]).T).T

        # Find the nearest point that is at least some r away
        dists = np.linalg.norm(pts_in_world - self.x0[None, :], axis=-1)
        points_far_away = pts_in_world[dists > self.r]
        pt_to_publish = points_far_away[0]

        # CREATING MESSAGE
        t_now = rospy.Time.now()

        # Variables to publish
        pos_msg = PointStamped()
        att_msg = QuaternionStamped()

        # Position
        pos_msg.header.stamp = t_now
        pos_msg.header.frame_id = "map"

        pos_msg.point.x = pt_to_publish[0]
        pos_msg.point.y = pt_to_publish[1]
        pos_msg.point.z = pt_to_publish[2]

        # Attitude
        att_msg.header.stamp = t_now
        att_msg.header.frame_id = "map"
        
        att_msg.quaternion.w = 1.
        att_msg.quaternion.x = 0.
        att_msg.quaternion.y = 0.
        att_msg.quaternion.z = 0.

        # Publish
        self.pos_pub.publish(pos_msg)
        self.att_pub.publish(att_msg)
   
        if np.linalg.norm(self.x0[:3] - self.xf[:3]) < 0.1:
            print("Mission Complete")
            rospy.signal_shutdown("Reached Goal")

    def update_current_pos(self, msg):
        self.x0 = np.array([msg.point.x, msg.point.y, msg.point.z])

if __name__ == '__main__':
    rospy.init_node('catnips_node')
    sub_topic = '/drone2/mavros/vision_pose/'
    catnips = CATNIPS_ROS(sub_topic=sub_topic)

    rospy.Timer(rospy.Duration(catnips.dt), catnips.traj_out)
    rospy.spin()