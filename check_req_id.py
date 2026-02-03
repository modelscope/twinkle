from sys import argv
from twinkle.server.utils.state import get_server_state
import ray

ray.init(namespace="twinkle_cluster")
if __name__ == '__main__':
    # get first argument
    req_id = argv[1]
    state = get_server_state()

    res = state.get_future(req_id)
    print(res)