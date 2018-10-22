from terial.config import SUBSTANCES
from toolbox.images import visualize_map


def visualize_segment_map(segment_map):
    return visualize_map(segment_map,
                         bg_value=-1,
                         values=range(segment_map.max() + 1))


def visualize_substance_map(substance_map):
    return visualize_map(substance_map,
                         bg_value=SUBSTANCES.index('background'),
                         values=range(len(SUBSTANCES)-1))
