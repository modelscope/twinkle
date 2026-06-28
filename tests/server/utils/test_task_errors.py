from twinkle.server.utils.task_errors import task_error_payload


def test_task_error_payload_keeps_lora_traceback():
    error = 'Traceback...\nRuntimeError: No lora available for tenant session-default. Max loras: 3\n'

    assert task_error_payload(error) == {
        'error': error,
        'category': 'Server',
    }
