from traderl.env.env1 import Trade as Env1
from traderl.env.env2 import Trade as Env2
from traderl.env.env3 import Trade as Env3
from traderl.env.env4 import Trade as Env4
from traderl.env.env5 import Trade as Env5
from traderl.env.base import Trade as DiscreteEnv0
from traderl.env.discreteenv1 import Trade as DiscreteEnv1
from traderl.env.discreteenv2 import Trade as DiscreteEnv2

env = {"env1": Env1, "env2": Env2, "env3": Env3, "env4": Env4, "env5": Env5,
       "discrete_env0": DiscreteEnv0, "discrete_env1": DiscreteEnv1, "discrete_env2": DiscreteEnv2}
