'Twinkle TransformersModel bridge for the Qwen3-TTS two-stage forward and loss.'

from copy import copy

from twinkle import remote_function
from twinkle.infra import collect_tensor_dict
from twinkle.model.transformers import TransformersModel

from .forward import qwen3_tts_forward_step
from .input_processor import Qwen3TTSInputProcessor


class Qwen3TTSModel(TransformersModel):
    """Override `forward` 走 Qwen3-TTS 2 段编排 + stash 中间态。

    `calculate_loss` / `backward` / `clip_grad_norm` 等复用基类（保留梯度累积/clip/metric 抽象
    + 多 LoRA 透传）。cookbook 仍调 `model.forward_backward(inputs=batch)`，编排全在本类内。
    """

    def _construct_default_optimizer_group(self):
        # Keep current Twinkle's group/metrics lifecycle, but pass tokenizer
        # policy at construction (including groups created for new adapters).
        from twinkle.model.transformers.transformers import OptimizerGroup
        from .tokenization import local_template
        from .loss import Qwen3TTSLoss
        return OptimizerGroup(
            loss_instance=Qwen3TTSLoss(),
            template=local_template(self.tokenizer_id),
            processor=Qwen3TTSInputProcessor(device_mesh=self.device_mesh),
            _device_mesh=self.device_mesh,
        )

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict)
    def forward(self, *, inputs, **kwargs):
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        # task 不用 causal_lm，但 pop 掉避免传给下游 qwen3_tts_forward_step
        kwargs.pop('task', None)
        kwargs.pop('temperature', 1.0)
        kwargs.pop('sampling_masks', None)
        kwargs.pop('return_logits', False)
        optimizer_config = self.optimizer_group[adapter_name]
        self._lazy_wrap_model()
        if not inputs:
            raise ValueError('inputs empty, check your DataLoader outputs')
        self.model.train()

        # 1. Trajectory / List[Trajectory] → InputFeature（若未编码）
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (
            isinstance(inputs, list) and self._not_encoded(inputs[0])
        ):
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs)

        # 2. 专用 processor 保留 TTS 字段和 batch 维度。
        processor = optimizer_config.processor
        if not isinstance(processor, Qwen3TTSInputProcessor):
            raise TypeError('Call set_processor(Qwen3TTSInputProcessor) before TTS forward')
        inputs = processor(
            inputs,
            sp_strategy=self.sp_strategy,
            model=self.model,
            hf_config=self.hf_config,
            enable_sp=getattr(self, '_enable_sp', False),
        )

        # 3. Qwen3-TTS 2 段前向编排（替换基类的 `self.model(**inputs)`）
        #    产出中间态 dict：talker_loss / sub_talker_logits / sub_talker_labels /
        #    target_speaker_embedding。
        outputs = qwen3_tts_forward_step(self.model, inputs)

        # 4. stash 中间态进 train_status.outputs，供 Qwen3TTSLoss 从 outputs 读（calculate_loss 复用基类）
        status = optimizer_config.train_status
        status.inputs = inputs
        status.outputs = outputs
        status.forward_kwargs = kwargs
        status.loss_value = 0
        return copy(outputs)
