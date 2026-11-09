# policy_interfaces/msg/PolyTrajectoryMsg.msg
std_msgs/Header header          # header.stamp = t0 (tau=0 of segment 0), header.frame_id = world frame
uint32 id                       # increments per (re)plan
uint8 n_coeff                   # = 8, checked on recieve
float64[] durations             # (S)
float64[] coeffs                # (S * n_coeff * 3), row-majer [s][n][d] - same as PolyTrajectory.coeffs

