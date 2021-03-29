import numpy as np
from .src.agent import OwnAgent
from .src.StateProcessor import StateProcessor
from .src.parameters import OPTIONS, get_parameters


def setup(self):
    np.random.seed()

    self.logger.debug('Successfully entered setup code')

    AGENT_CONFIG, _, N = get_parameters()

    self.agent = OwnAgent(**AGENT_CONFIG, training=self.train)
    self.state_processor = StateProcessor(N)


def act(self, game_state):

    if game_state['step'] == 1:
        self.state_processor.set_up_for_new_round()

    state, extra_state = self.state_processor.get_feature_space(game_state)

    action = self.agent.act(state, extra_state)

    action = OPTIONS[action]

    game_state['field'] = None
    game_state['explosion_map'] = None
    self.logger.debug(f'{game_state}: {action}')

    return action
