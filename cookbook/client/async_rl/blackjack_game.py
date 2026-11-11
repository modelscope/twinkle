# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared Blackjack task contract and the OpenSpiel command run in a sandbox."""
import argparse
from pathlib import Path

import json


class BlackjackTask:
    """Game instructions and tools shared by the local and E2B environments."""

    system_prompt = (
        'Play Blackjack. Use play(action="hit") to draw or play(action="stand") to stop. '
        'Try to get close to 21 without going over; an ace counts as 1 or 11. '
        'After the game ends, give a short final answer without more tool calls.')

    def tools(self):
        # Return fresh schemas so an episode cannot mutate another's tool contract.
        return [{
            'type': 'function',
            'function': {
                'name': 'play', 'description': 'Take the next Blackjack action.',
                'parameters': {'type': 'object', 'properties': {
                    'action': {'type': 'string', 'enum': ['hit', 'stand']}}, 'required': ['action']},
            },
        }]


def play(seed, actions):
    from open_spiel.python.rl_environment import ChanceEventSampler, Environment

    game = Environment('blackjack', chance_event_sampler=ChanceEventSampler(seed))
    state = game.reset()
    for action in actions:
        if state.last():
            raise ValueError('game is already over')
        state = game.step([{'hit': 0, 'stand': 1}[action]])
    cards = state.observations['info_state'][0]
    if len(cards) != 189:
        raise RuntimeError('unexpected OpenSpiel Blackjack observation')
    ranks = ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K')
    player = [ranks[i % 13] for i in range(52) if cards[85 + i]]
    dealer = [ranks[i % 13] for i in range(52) if cards[137 + i]]
    return {
        'observation': f'Your cards: {player}. Dealer visible cards: {dealer}. Game over: {state.last()}.',
        'reward': float(state.rewards[0]) if state.rewards else 0.0,
        'done': state.last(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['reset', 'hit', 'stand'])
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    path = Path('game.json')
    if args.action == 'reset':
        if args.seed is None:
            parser.error('reset requires --seed')
        history = {'seed': args.seed, 'actions': []}
    else:
        history = json.loads(path.read_text())
        history['actions'].append(args.action)
    # Replay a tiny hand from its seed so separate commands share exactly one game.
    result = play(history['seed'], history['actions'])
    path.write_text(json.dumps(history))
    print(json.dumps(result))  # noqa: T201 - JSON response consumed by the client adapter.


if __name__ == '__main__':
    main()
