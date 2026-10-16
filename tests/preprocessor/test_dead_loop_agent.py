from twinkle_agentic.preprocessor.dead_loop_filter import DeadLoopFilter


def _row(*assistant_texts):
    msgs = [{'role': 'user', 'content': 'go'}]
    for i, t in enumerate(assistant_texts):
        msgs.append({
            'role': 'assistant',
            'content': t,
            'tool_calls': '[{"id":"1","type":"function","function":{"name":"x","arguments":"{}"}}]' if i == 0 else '',
        })
        if i == 0:
            msgs.append({'role': 'tool', 'content': 'ok', 'tool_call_id': '1'})
    return {'messages': msgs}


def test_agent_requires_two_stuck_turns():
    f = DeadLoopFilter(agent_min_stuck_turns=2)
    stuck = 'wait wait no actually hmm no wait oh wait i was wrong'
    kept, dropped = f([_row(stuck, 'ok reply')])
    assert len(kept) == 1 and not dropped


def test_agent_drops_on_two_stuck_turns():
    f = DeadLoopFilter(agent_min_stuck_turns=2)
    stuck = 'wait wait no actually hmm no wait oh wait i was wrong'
    kept, dropped = f([_row(stuck, stuck)])
    assert not kept and len(dropped) == 1
