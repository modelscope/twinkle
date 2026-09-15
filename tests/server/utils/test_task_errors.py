from twinkle.server.utils.task_errors import error_payload_from_stored, task_error_payload
from twinkle_client.types.errors import ErrorCategory


def test_task_error_payload_builds_error_payload_dict():
    error = 'RuntimeError: No lora available for tenant session-default. Max loras: 3'

    payload = task_error_payload(error, request_id='req_1', error_code=500)

    assert payload['error'] == error
    assert payload['category'] == ErrorCategory.Server.value
    assert payload['error_code'] == 500
    assert payload['request_id'] == 'req_1'
    assert 'traceback' not in payload


def test_task_error_payload_user_category_drops_traceback():
    payload = task_error_payload(
        'bad input', request_id='req_2', error_code=400, category=ErrorCategory.User, traceback_text='Traceback...')

    assert payload['category'] == ErrorCategory.User.value
    assert 'traceback' not in payload


def test_error_summary_is_single_line():
    payload = task_error_payload(
        'RuntimeError: boom\n  File "/server/path.py", line 1', request_id='req-lines')
    assert payload['error'] == 'RuntimeError: boom'


def test_error_payload_from_stored_backfills_two_field_legacy():
    stored = {'error': 'boom', 'category': 'Server'}

    payload = error_payload_from_stored(stored, request_id='req_3')

    assert payload.error_code == 500
    assert payload.request_id == 'req_3'
