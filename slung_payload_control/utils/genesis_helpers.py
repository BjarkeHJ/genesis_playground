import logging
import math

import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat

from components.flight_control.airframe import Rotor, Airframe

class AssetLogFilter(logging.Filter):
    # Drops Genesis' inertia-estimate messages for the massless prop/attach links (authored mass 0 is kept, verified)
    PATTERNS = ("dubious mass", "not watertight", "collision geoms can not be found")

    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in self.PATTERNS)

def silence_asset_warnings():
    # Call after gs.init (the Genesis logger exists from then on). Idempotent
    if not any(isinstance(f, AssetLogFilter) for f in gs.logger._logger.filters):
        gs.logger.addFilter(AssetLogFilter())

def airframe_from_genesis(drone, twr_max, payload_mass=0.0):
    # Built Genesis drone -> Airframe for FlightController.bind (nominal values, read from env 0).
    # Rotors: propeller positions about the drone's center of mass in body FRD (allocator convention), yaw moment ratio
    # signed by spin. rpm_max: the rpm at which all rotors together lift twr_max times the drone + payload_mass weight
    base = drone.base_link_idx - drone.link_start
    com = drone.get_links_pos([base], ref=gs.link_ref_frame.root_COM)[0, 0]
    props = drone.get_links_pos(drone.propellers_idx - drone.link_start)[0]
    q_inv = inv_quat(drone.get_quat()[0]).expand(len(props), 4)
    r = transform_by_quat(props - com, q_inv) # (n, 3) body FLU

    km = drone.KM / drone.KF
    rotors = tuple(Rotor(px=x, py=-y, km=-s * km) for (x, y, _), s in zip(r.tolist(), drone.propellers_spin.tolist())) # FLU -> FRD

    gravity = abs(drone.scene.sim.options.gravity[2])
    total_mass = drone.get_mass().flatten()[0].item() + payload_mass
    max_motor_thrust = twr_max * total_mass * gravity / drone.n_propellers
    return Airframe(rotors=rotors, rpm_max=math.sqrt(max_motor_thrust / drone.KF))
