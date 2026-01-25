import unittest

from twinkle.data_format import Message, Trajectory
from twinkle.template import Template
from twinkle.hub import HubOperation


class TestMMModel(unittest.TestCase):

    def test_mm(self):
        model_dir = HubOperation.download_model('ms://Qwen/Qwen3-VL-2B-Instruct')
        template = Template(model_dir)
        messages = [
            Message(
                role='user',
                content='how are you',
            ),
            Message(
                role='assistant',
                content='fine',
            ),
        ]
        trajectory = Trajectory(messages=messages)
        print(template.batch_encode([trajectory]))