"""What the machine can actually do, as opposed to what the layers above it ask for.

MEASURED BUG THIS EXISTS TO FIX (2026-08-01). The demo harnesses integrated the
controller's output straight into the state -- `v_new = u[0]` -- so the robot's speed
could change by any amount in one 0.1 s step. Measured on the commissioning route, all
three arms braked at **-4.5 to -5.8 m/s^2** against a platform limit of 1.2, while
accelerating correctly at +0.6. Every machine could therefore shed a speed limit the
instant it was told about it, which shortened every mission time and made a zone
boundary look like a step change in speed rather than something a vehicle has to
decelerate for.

The cause is a known interface defect, not solver noise: `v_max_cmd` enters the MPC as a
HARD bound `v_k <= v_max_cmd`, while the same program carries `|dv| <= a_max_mpc*dt`.
When the cap drops by more than one window's worth of deceleration the two constraints
cannot both hold, the program is infeasible, and IPOPT returns an infeasible iterate --
which the harness then applied as if it were a real command.

Two layers of defence, because they fix different halves of it:

* `reachable_cap` keeps the PROGRAM feasible by never asking for a cap the machine could
  not reach this step. This is the fix recorded against the defect.
* `apply_plant` enforces the physical limit on whatever finally comes out, so no layer --
  MPC, CBF, scanner or supervisor -- can command a speed change the vehicle cannot make.
  A safety filter that is right about the world but wrong about the vehicle still ends up
  wrong.

A protective stop is deliberately EXEMPT from `reachable_cap`. On a real machine the
safety controller cuts the drives directly rather than politely asking the planner, and
the protective field is sized on the service brake, so rate-limiting it to the planner's
comfortable deceleration would undersize the field. It still cannot beat `apply_plant`.
"""
from __future__ import annotations


def reachable_cap(v_max_cmd: float, v_prev: float, plat, emergency: bool = False):
    """The speed cap the MPC can actually honour this step, given where it is now.

    Without this the cap and the program's own deceleration limit contradict each other
    and the solve is infeasible. `emergency` passes a protective stop straight through.
    """
    if emergency:
        return float(v_max_cmd)
    floor = v_prev - plat.robot.a_max_mpc * plat.robot.dt
    return float(max(v_max_cmd, floor))


def apply_plant(v_cmd: float, v_prev: float, plat) -> float:
    """Speed the wheels actually reach, bounded by the platform's acceleration.

    `a_max_physical` is above both the MPC's planning limit and the CBF's service
    braking assumption, so nothing either of them legitimately plans is clipped here --
    only the impossible transients are.
    """
    step = plat.robot.a_max_physical * plat.robot.dt
    return float(min(max(v_cmd, v_prev - step), v_prev + step))
