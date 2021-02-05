import interface as bbox
from RLAlgo import SARSAEpisode, QEpisode


n_features = n_actions = -1

def prepare_bbox():
    global n_features, n_actions

    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()


def run_bbox(verbose=False):
    has_next = 1

    prepare_bbox()

    # CartPoleDemo(bbox)
    getInitialState = lambda: bbox.get_state().copy()
    doAction = lambda a: [not bbox.do_action(a), bbox.get_state()]
    getReward = lambda: bbox.get_score()
    actions_list = [0,1,2,3]
    QEpisode(getInitialState, doAction, getReward, actions_list, maxsteps=10000000)



    bbox.finish(verbose=1)


if __name__ == "__main__":
    run_bbox(verbose=0)
