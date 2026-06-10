STREAM_SENTINEL = '__STREAM_END__'


def stream_to_queue(sampler, queue, inputs, sampling_params=None, adapter_name='', adapter_path=None):
    """Push streaming deltas from *sampler* to a cross-process Ray queue.

    Works with any object that exposes a ``sample_stream`` iterator.
    """
    try:
        for delta, reason in sampler.sample_stream(inputs, sampling_params, adapter_name, adapter_path):
            queue.put((delta, reason))
    except Exception as e:
        queue.put(e)
    finally:
        queue.put(STREAM_SENTINEL)
