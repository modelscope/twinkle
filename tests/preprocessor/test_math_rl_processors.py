from twinkle.preprocessor import DAPOMathProcessor


def test_dapo_math_processor_preserves_prompt_and_ground_truth():
    row = {
        'prompt': [{
            'role': 'user',
            'content': 'Solve the problem and put the answer in a box.',
        }],
        'reward_model': {
            'ground_truth': '34',
            'style': 'rule-lighteval/MATH_v2',
        },
    }

    trajectory = DAPOMathProcessor().preprocess(row)

    assert trajectory['messages'] == row['prompt']
    assert trajectory['user_data'] == [('ground_truth', '34')]
