from twinkle.loss.base import Loss


class GRPOLoss(Loss):

    def __call__(self, inputs, outputs, ref_logits):
