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
    else:
        raise ValueError('collsion_checker type {} not recognized'.format(base_type))


def GUI_factory_base(base_type):
    if base_type == 'matplotlib':
        from robot_planning.environment.guis import MatplotlibGui
        return MatplotlibGui()
    else:
        raise ValueError('GUI type {} not recognized'.format(base_type))


def goal_checker_factory_base(base_type):
    if base_type == 'bicycle_model_goal_checker':
        from robot_planning.environment.goal_checker import BicycleModelGoalChecker
        return BicycleModelGoalChecker()
    else:
        raise ValueError('goal_checker type {} not recognized'.format(base_type))



