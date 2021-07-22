def dynamics_factory_base(base_type):
    if base_type == 'bicycle_dynamics':
        from robot_planning.environment.dynamics.bicycle_dynamics import BicycleDynamics
        return BicycleDynamics()
    else:
        raise ValueError('dynamics type {} not recognized'.format(base_type))


def cost_evaluator_factory_base(base_type):
    if base_type == 'quadratic_cost':
        from robot_planning.environment.cost_evaluators import QuadraticCostEvaluator
        return QuadraticCostEvaluator()
    else:
        raise ValueError('cost_evaluator type {} not recognized'.format(base_type))


def collision_checker_factory_base(base_type):
    if base_type == 'bicycle_model_collision_checker':
        from robot_planning.environment.collision_checker import BicycleModelCollisionChecker
        return BicycleModelCollisionChecker()
    elif base_type == 'point_collision_checker':
        from robot_planning.environment.collision_checker import PointCollisionChecker
        return PointCollisionChecker()
    else:
        raise ValueError('collsion_checker type {} not recognized'.format(base_type))


def renderer_factory_base(base_type):
    if base_type == 'MPPImatplotlib':
        from robot_planning.environment.renderers import MPPIMatplotlibRenderer
        return MPPIMatplotlibRenderer()
    else:
        raise ValueError('Visualizer type {} not recognized'.format(base_type))


def goal_checker_factory_base(base_type):
    if base_type == 'state_space_goal_checker':
        from robot_planning.environment.goal_checker import StateSpaceGoalChecker
        return StateSpaceGoalChecker()
    else:
        raise ValueError('goal_checker type {} not recognized'.format(base_type))


def robot_factory_base(base_type):
    if base_type == 'simulated_robot':
        from robot_planning.environment.robots.simulated_robot import SimulatedRobot
        return SimulatedRobot()
    else:
        raise ValueError('robot type {} is not recognized'.format(base_type))


def kinematics_factory_base(base_type):
    if base_type == 'bicycle_model_kinematics':
        from robot_planning.environment.kinematics.simulated_kinematics import BicycleModelKinematics
        return BicycleModelKinematics()
    elif base_type == 'point_kinematics':
        from robot_planning.environment.kinematics.simulated_kinematics import PointKinematics
        return PointKinematics()
    else:
        raise ValueError('kinematics type {} is not recognized'.format(base_type))


def controller_factory_base(base_type):
    if base_type == 'MPPI':
        from robot_planning.controllers.MPPI.MPPI import MPPI
        return MPPI()
    else:
        raise ValueError('controller type {} is not recognized'.format(base_type))


def stochastic_trajectories_sampler_factory_base(base_type):
    if base_type == 'MPPI_stochastic_trajectories_sampler':
        from robot_planning.controllers.MPPI.stochastic_trajectories_sampler import MPPIStochasticTrajectoriesSampler
        return MPPIStochasticTrajectoriesSampler()
    else:
        raise ValueError('stochastic_trajectories_sampler type {} is not recognized'.format(base_type))


def noise_sampler_factory_base(base_type):
    if base_type == 'gaussian_noise_sampler':
        from robot_planning.controllers.MPPI.noise_sampler import GaussianNoiseSampler
        return GaussianNoiseSampler()
    else:
        raise ValueError('noise_sampler type {} is not recognized'.format(base_type))