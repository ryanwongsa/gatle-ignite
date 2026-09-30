"""ignite's Accuracy reduces across ranks; a metric summing python floats is rank-local."""

from ignite.metrics import Accuracy

from gatle_ignite import get_value


def get_metric(engine_type, info, src_name="logits", tgt_name=("targets", "labels"), **kwargs):
    def output_transform(output):
        return get_value(output["y_pred"], src_name), get_value(output["target"], tgt_name)

    return Accuracy(output_transform=output_transform, **kwargs)
