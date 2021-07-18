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


def GUI_factory_base(base_type):
    if base_type == 'matplotlib':
        from robot_planning.environment.guis import MatplotlibGui
        return MatplotlibGui()
    else:
        raise ValueError('GUI type {} not recognized'.format(base_type))


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
        raise ValueError('robot_name {} is not recognized'.format(base_type))


def kinematics_factory_base(base_type):
    if base_type == 'bicycle_model_kinematics':
        from robot_planning.environment.kinematics.simulated_kinematics import BicycleModelKinematics
        return BicycleModelKinematics()
    elif base_type == 'point_kinematics':
        from robot_planning.environment.kinematics.simulated_kinematics import PointKinematics
        return PointKinematics()
    else:
        raise ValueError('kinematics name {} is not recognized'.format(base_type))


def rl_agent_factory_base(base_type):
    if base_type == 'maddpg':
        from robot_planning.trainers.rl_agents.maddpg_agent import MADDPG_Agent
        return MADDPG_Agent()
    else:
        raise ValueError('agent type {} is not recognized'.format(base_type))


def observer_base(base_type):
    if base_type == 'full_observation':
        from robot_planning.environment.observer import FullStateObserver
        return FullStateObserver()
    if base_type == 'local_state_observation':
        from robot_planning.environment.observer import LocalStateObserver
        return LocalStateObserver()
    else:
        raise ValueError('observer type {} is not recognized'.format(base_type))
