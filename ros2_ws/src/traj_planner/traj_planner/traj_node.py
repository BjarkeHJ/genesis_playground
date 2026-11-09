import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from policy_interfaces.msg import PolyTrajectoryMsg
from traj_planner.route import Route
from traj_planner.trajectory import PolyTrajectory
from traj_planner.min_snap_solver import MinSnapSolver

class TrajectoryNode(Node):
    def __init__(self):
        super().__init__("trajectory_planner_node")
        self.get_logger().info("TrajectoryNode has been created")

        self.qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(PolyTrajectoryMsg, "planner/trajectory", self.qos)

        self.solver = MinSnapSolver()

    def publish_trajectory():
        msg = PolyTrajectoryMsg()

    def _build_traj():
        ...

    def _load_params(self):
        ...




# Spin node
def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()