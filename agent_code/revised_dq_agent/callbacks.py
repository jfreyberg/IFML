import numpy as np
from .src.agent import OwnAgent
from .src.StateProcessor import StateProcessor
from .src.parameters import OPTIONS, INVERSE_OPTIONS, get_parameters
from .src.rule_based_agent import RuleBasedAgent


def setup(self):
    np.random.seed()

    self.logger.debug('Successfully entered setup code')

    AGENT_CONFIG, _, N = get_parameters()

    self.agent = OwnAgent(**AGENT_CONFIG, training=self.train)
    self.state_processor = StateProcessor(N)

    if self.train:
        self.rule_based_agent = RuleBasedAgent()


def act(self, game_state):

    if game_state['step'] == 1:
        self.state_processor.set_up_for_new_round()

    rulebased_action = None
    if self.train:
        rulebased_action = self.rule_based_agent.act_rule_based(game_state)

    if rulebased_action is None:
        rulebased_action = 'WAIT'
    rulebased_action = INVERSE_OPTIONS[rulebased_action]

    state, extra_state = self.state_processor.get_feature_space(game_state)

    action = self.agent.act(state, extra_state, rulebased_action)

    action = OPTIONS[action]

    game_state['field'] = None
    game_state['explosion_map'] = None
    self.logger.debug(f'{game_state}: {action}')

    if not self.train and not self.agent.evaluate_model:
        print(action)

    return action
