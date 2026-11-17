import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Duration
from rclpy.parameter import Parameter

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import numpy as np

from policy_interfaces.msg import PolyTrajectoryMsg
from traj_planner.route import Route
from traj_planner.min_snap_solver import MinSnapSolver, N_COEFF

class TrajectoryNode(Node):
    def __init__(self):
        super().__init__("trajectory_planner_node")

        self.qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.pub = self.create_publisher(
            PolyTrajectoryMsg, 
            "planner/trajectory", 
            self.qos
        )
        self.pub_viz = self.create_publisher(
            Path,
            "planner/trajectory_viz",
            self.qos
        )
        self.do_viz = True
        self.viz_dt = 0.05 # [s]

        self.solver = MinSnapSolver()
        self.waypoints = None
        self.traj = None
        self.start_delay = 3.0
        self.plan_id = 0
        self.frame_id = ""

        self._load_params()
        
        self.create_service(Trigger, "~/start", self._on_start)

        if len(self.waypoints):
            self.set_route(self.waypoints)

        self.get_logger().info("TrajectoryNode has been created.")

    def set_route(self, waypoints):
        route = Route(waypoints, self.vel_des, self.acc_max, self.corridor)
        self.traj = self.solver.solve(route).enforce_limits(self.acc_max, self.jerk_max)
        self.get_logger().info(f"Solved: {self.traj.metrics()}")
        if self.do_viz:
            self._publish_viz()

    def _on_start(self, request, response):
        if self.traj is None:
            response.success = False
            response.message = "no trajectory solved"
            return response
        self._publish(self.get_clock().now() + Duration(seconds=self.start_delay))
        response.success = True
        return response
    
    def _publish(self, t0):
        self.plan_id += 1
        msg = PolyTrajectoryMsg()
        msg.header.stamp = t0.to_msg()
        msg.header.frame_id = self.frame_id
        msg.id = self.plan_id
        msg.n_coeff = N_COEFF
        msg.durations = self.traj.durations.tolist()
        msg.coeffs = self.traj.coeffs.reshape(-1).tolist()
        self.pub.publish(msg)
        self.get_logger().info(f"Published PolyTrajectoryMsg id: {msg.id}")

    def _load_params(self):
        # Params (align with trained TrajectoryConfig)
        self.declare_parameter("desired_velocity", 2.0)
        self.declare_parameter("max_acceleration", 6.0)
        self.declare_parameter("max_jerk", 8.0)
        self.declare_parameter("route_corridor", 0.5)
        self.vel_des = self.get_parameter("desired_velocity").value
        self.acc_max = self.get_parameter("max_acceleration").value
        self.jerk_max = self.get_parameter("max_jerk").value
        self.corridor = self.get_parameter("route_corridor").value

        self.declare_parameter("waypoints", Parameter.Type.DOUBLE_ARRAY)
        flat = np.asarray(self.get_parameter("waypoints").value, dtype=np.float64)
        if flat.size == 0 or flat.size % 3 != 0:
            raise ValueError(f"Waypoints must be x,y,z triples, got {flat.size} values")
        self.waypoints = flat.reshape(-1, 3)

        self.declare_parameter("frame_id", "world")
        self.frame_id = self.get_parameter("frame_id").value
        
        self.get_logger().info(f"Route Parameters: vel_des={self.vel_des}, acc_max={self.acc_max}, jerk_max={self.jerk_max}, corridor={self.corridor}")
        self.get_logger().info(f"Route waypoints: {self.waypoints.shape[0]}")
        self.get_logger().info(f"Frame: {self.frame_id}")

    def _publish_viz(self):
        _, pos = self.traj.sample(self.viz_dt)
        path = Path()
        path.header.frame_id = self.frame_id
        for x, y, z in pos.tolist():
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.pub_viz.publish(path)

# Spin node
def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()