def dynamics_factory_base(base_type):
    if base_type == 'bicycle_dynamics':
        from robot_planning.environment.dynamics.bicycle_dynamics import BicycleDynamics
        return BicycleDynamics()
    if base_type == 'point_dynamics':
        from robot_planning.environment.dynamics.point_dynamics import PointDynamics
        return PointDynamics()
    if base_type == 'abstract_dynamics':
        from robot_planning.environment.dynamics.abstract_dynamics import AbstractDynamics
        return AbstractDynamics()
    if base_type == 'autorally_dynamics':
        from robot_planning.environment.dynamics.autorally_dynamics.autorally_dynamics import AutoRallyDynamics
        return AutoRallyDynamics()
    else:
        raise ValueError('dynamics type {} not recognized'.format(base_type))


def cost_evaluator_factory_base(base_type):
    if base_type == 'quadratic_cost':
        from robot_planning.environment.cost_evaluators import QuadraticCostEvaluator
        return QuadraticCostEvaluator()
    if base_type == 'terminal_cost':
        from robot_planning.environment.cost_evaluators import TerminalCostEvaluator
        return TerminalCostEvaluator()
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
    if base_type == 'Envmatplotlib':
        from robot_planning.environment.renderers import EnvMatplotlibRenderer
        return EnvMatplotlibRenderer()
    if base_type == 'autorally_matplotlib':
        from robot_planning.environment.renderers import AutorallyMatplotlibRenderer
        return AutorallyMatplotlibRenderer()
    else:
        raise ValueError('Visualizer type {} not recognized'.format(base_type))


def goal_checker_factory_base(base_type):
    if base_type == 'state_space_goal_checker':
        from robot_planning.environment.goal_checker import StateSpaceGoalChecker
        return StateSpaceGoalChecker()
    if base_type == 'flex_state_space_goal_checker':
        from robot_planning.environment.goal_checker import FlexStateSpaceGoalChecker
        return FlexStateSpaceGoalChecker()
    if base_type == 'position_goal_checker':
        from robot_planning.environment.goal_checker import PositionGoalChecker
        return PositionGoalChecker()
    else:
        raise ValueError('goal_checker type {} not recognized'.format(base_type))


def robot_factory_base(base_type):
    if base_type == 'simulated_robot':
        from robot_planning.environment.robots.simulated_robot import SimulatedRobot
        return SimulatedRobot()
    if base_type == "abstract_robot":
        from robot_planning.environment.robots.abstract_robot import AbstractRobot
        return AbstractRobot()
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
    elif base_type == 'CSSMPC':
        from robot_planning.controllers.CSSMPC.CSSMPC import CSSMPC
        return CSSMPC()
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


def rl_agent_factory_base(base_type):
    if base_type == 'maddpg':
        from robot_planning.trainers.rl_agents.maddpg_agent import MADDPG_Agent
        return MADDPG_Agent()
    else:
        raise ValueError('agent type {} is not recognized'.format(base_type))


def observer_factory_base(base_type):
    if base_type == 'full_state_observer':
        from robot_planning.environment.observer import FullStateObserver
        return FullStateObserver()
    if base_type == 'local_state_observer':
        from robot_planning.environment.observer import LocalStateObserver
        return LocalStateObserver()
    else:
        raise ValueError('observer type {} is not recognized'.format(base_type))


def configs_generator_factory_base(base_type):
    if base_type == 'MPPI_configs_generator':
        from robot_planning.batch_experimentation.configs_generator import MPPIConfigsGenerator
        return MPPIConfigsGenerator()
    else:
        raise ValueError('config_generator type {} is not recognized'.format(base_type))


def logger_factory_base(base_type):
    if base_type == 'MPPI_logger':
        from robot_planning.batch_experimentation.loggers import MPPILogger
        return MPPILogger()
    else:
        raise ValueError('logger type {} is not recognized'.format(base_type))


def dynamics_linearizer_factory_base(base_type):
    if base_type == 'numpy_dynamics_linearizer':
        from robot_planning.environment.dynamics_linearizer import NumpyDynamicsLinearizer
        return NumpyDynamicsLinearizer()
    else:
        raise ValueError('dynamics_linearizer type {} is not recognized'.format(base_type))
