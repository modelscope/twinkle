import ray
# cookbook/grpo/lora.py
import os
import torch

# -----------------------------------------------------------------------------
# AttrDict: dict with attribute access (supports both .keys() and .labels usage)
# -----------------------------------------------------------------------------
class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, get_device_placement
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.infra import DeviceGroup, remote_function, remote_class
from twinkle.model import TransformersModel
from twinkle.processor import GRPOLossProcessor
from twinkle.reward import MathReward

# -----------------------------------------------------------------------------
# Minimal shim for MathReward: it expects trajectory.messages[-1].content
# Our sampler currently returns list[dict] like {"prompt": ..., "response": ...}.
# This shim keeps Goal A minimal loop moving.
# -----------------------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content: str):
        import os, torch
        print("[ActorGroup init] pid="+str(os.getpid())+" CUDA_VISIBLE_DEVICES="+str(os.environ.get("CUDA_VISIBLE_DEVICES")))
        print("[ActorGroup init] torch.cuda.is_available="+str(torch.cuda.is_available())+" count="+str(torch.cuda.device_count()))
        if torch.cuda.is_available():
            names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            print("[ActorGroup init] current="+str(torch.cuda.current_device())+" names="+str(names))
        import os, torch
        print("[ActorGroup init] pid=%s CUDA_VISIBLE_DEVICES=%s" % (os.getpid(), os.environ.get("CUDA_VISIBLE_DEVICES")))
        print("[ActorGroup init] torch.cuda.is_available=%s count=%s" % (torch.cuda.is_available(), torch.cuda.device_count()))
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print("[ActorGroup init] current=%s name0=%s" % (torch.cuda.current_device(), torch.cuda.get_device_name(0)))
        self.content = content

class _FakeTrajectory:
    def __init__(self, prompt: str, response: str):
        import os, torch
        print("[ActorGroup init] pid="+str(os.getpid())+" CUDA_VISIBLE_DEVICES="+str(os.environ.get("CUDA_VISIBLE_DEVICES")))
        print("[ActorGroup init] torch.cuda.is_available="+str(torch.cuda.is_available())+" count="+str(torch.cuda.device_count()))
        if torch.cuda.is_available():
            names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            print("[ActorGroup init] current="+str(torch.cuda.current_device())+" names="+str(names))
        import os, torch
        print("[ActorGroup init] pid=%s CUDA_VISIBLE_DEVICES=%s" % (os.getpid(), os.environ.get("CUDA_VISIBLE_DEVICES")))
        print("[ActorGroup init] torch.cuda.is_available=%s count=%s" % (torch.cuda.is_available(), torch.cuda.device_count()))
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print("[ActorGroup init] current=%s name0=%s" % (torch.cuda.current_device(), torch.cuda.get_device_name(0)))
        # only what MathReward.calculate() needs
        self.messages = [_FakeMessage(prompt), _FakeMessage(response)]
        self.prompt = prompt
        self.response = response

def _wrap_trajectories_for_reward(trajs):
    # If already Twinkle trajectories (have .messages), return as-is
    if trajs and hasattr(trajs[0], "messages"):
        return trajs
    # If sampler returned dicts, wrap them
    out = []
    for t in trajs:
        if isinstance(t, dict):
            resp = t.get("response", None)
            if resp is None:
                for k in ("answer", "output", "solution", "target", "label", "ground_truth"):
                    if k in t and t[k] is not None:
                        resp = t[k]
                        break
            out.append(_FakeTrajectory(str(t.get("prompt", "")), str(resp if resp is not None else "")))
        else:
            # fallback: stringify
            out.append(_FakeTrajectory("", str(t)))
    return out

# -----------------------------------------------------------------------------
# Minimal shim for GRPOLossProcessor: it expects inputs.labels (and usually input_ids/attention_mask).
# DataLoader here yields python list/dict; convert to an attribute-style object.
# -----------------------------------------------------------------------------
from types import SimpleNamespace

def _to_attr_inputs(batch, device=None):
    """Convert list/dict batch to an object with attribute access: inputs.labels, inputs.input_ids, ..."""
    if batch is None:
        return SimpleNamespace()

    # If already attribute-style
    if hasattr(batch, "__dict__") and hasattr(batch, "labels"):
        return batch

    # dict -> namespace
    if isinstance(batch, dict):
        d = dict(batch)
        # Ensure labels exists
        if "labels" not in d:
            if "label" in d:
                d["labels"] = d["label"]
            elif "input_ids" in d:
                # minimal fallback: labels = input_ids (not semantically perfect but keeps loop alive)
                try:
                    d["labels"] = d["input_ids"].clone()
                except Exception:
                    d["labels"] = d["input_ids"]
        ns = SimpleNamespace(**d)
        return ns

    # list[dict] -> collate into dict of tensors/lists then namespace
    if isinstance(batch, list) and batch and isinstance(batch[0], dict):
        keys = set(batch[0].keys())
        for x in batch[1:]:
            keys &= set(x.keys())
        keys = list(keys)

        collated = {}
        for k in keys:
            vals = [x.get(k) for x in batch]
            v0 = vals[0]
            # torch tensor: stack if same shape
            if torch.is_tensor(v0):
                try:
                    collated[k] = torch.stack(vals, dim=0)
                except Exception:
                    collated[k] = vals
            else:
                collated[k] = vals

        # Ensure labels exists
        if "labels" not in collated:
            if "label" in collated:
                collated["labels"] = collated["label"]
            elif "input_ids" in collated:
                try:
                    collated["labels"] = collated["input_ids"].clone()
                except Exception:
                    collated["labels"] = collated["input_ids"]

        # Move tensors to device if given
        if device is not None:
            for k, v in list(collated.items()):
                if torch.is_tensor(v):
                    collated[k] = v.to(device)

        return SimpleNamespace(**collated)

    # fallback: wrap everything as text-like labels
    return SimpleNamespace(labels=str(batch))


# -----------------------------------------------------------------------------
# Minimal shim for GRPOLossProcessor: it expects inputs.labels (and usually input_ids/attention_mask).
# DataLoader here yields python list/dict; convert to an attribute-style object.
# -----------------------------------------------------------------------------
from types import SimpleNamespace

def _to_attr_inputs(batch, device=None):
    """Convert list/dict batch to an object with attribute access: inputs.labels, inputs.input_ids, ..."""
    if batch is None:
        return SimpleNamespace()

    # If already attribute-style
    if hasattr(batch, "__dict__") and hasattr(batch, "labels"):
        return batch

    # dict -> namespace
    if isinstance(batch, dict):
        d = dict(batch)
        # Ensure labels exists
        if "labels" not in d:
            if "label" in d:
                d["labels"] = d["label"]
            elif "input_ids" in d:
                # minimal fallback: labels = input_ids (not semantically perfect but keeps loop alive)
                try:
                    d["labels"] = d["input_ids"].clone()
                except Exception:
                    d["labels"] = d["input_ids"]
        ns = SimpleNamespace(**d)
        return ns

    # list[dict] -> collate into dict of tensors/lists then namespace
    if isinstance(batch, list) and batch and isinstance(batch[0], dict):
        keys = set(batch[0].keys())
        for x in batch[1:]:
            keys &= set(x.keys())
        keys = list(keys)

        collated = {}
        for k in keys:
            vals = [x.get(k) for x in batch]
            v0 = vals[0]
            # torch tensor: stack if same shape
            if torch.is_tensor(v0):
                try:
                    collated[k] = torch.stack(vals, dim=0)
                except Exception:
                    collated[k] = vals
            else:
                collated[k] = vals

        # Ensure labels exists
        if "labels" not in collated:
            if "label" in collated:
                collated["labels"] = collated["label"]
            elif "input_ids" in collated:
                try:
                    collated["labels"] = collated["input_ids"].clone()
                except Exception:
                    collated["labels"] = collated["input_ids"]

        # Move tensors to device if given
        if device is not None:
            for k, v in list(collated.items()):
                if torch.is_tensor(v):
                    collated[k] = v.to(device)

        return SimpleNamespace(**collated)

    # fallback: wrap everything as text-like labels
    return SimpleNamespace(labels=str(batch))


# -----------------------------------------------------------------------------
# 0) Offline/local model path
# -----------------------------------------------------------------------------
LOCAL_MODEL_PATH = (
    os.environ.get("TWINKLE_MODEL_PATH")
    or os.environ.get("LOCAL_MODEL_PATH")
    or "/root/torch/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
)
assert os.path.isdir(LOCAL_MODEL_PATH), f"LOCAL_MODEL_PATH not found: {LOCAL_MODEL_PATH}"

# -----------------------------------------------------------------------------
# 1) Goal A: 2 GPUs total
# -----------------------------------------------------------------------------
device_groups = [
    DeviceGroup(name="actor", ranks=[0], device_type="GPU"),
    DeviceGroup(name="ref",   ranks=[1], device_type="GPU"),
]

actor_device_mesh = DeviceMesh(device_type="cuda", mesh=[0], mesh_dim_names=("dp",))
ref_device_mesh   = DeviceMesh(device_type="cuda", mesh=[1], mesh_dim_names=("dp",))


def create_dataset():
    dataset = Dataset(DatasetMeta("ms://modelscope/competition_math"))
    dataset.set_template("Qwen3Template", model_id=LOCAL_MODEL_PATH)
    dataset.map("CompetitionMathProcessor")
    dataset.check(batched=True)
    return dataset


@remote_class()
@ray.remote(num_gpus=1)
class ActorGroup:
    """
    Goal A: no vLLM. Sampling + training both inside this actor using transformers.
    """

    def __init__(
        self,
        lora_config: LoraConfig,
        adapter_name: str = "default",
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kwargs,
    ):
        self.adapter_name = adapter_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        # IMPORTANT: inside remote actor, don't pass remote_group again
        self.model = TransformersModel(
            pretrained_model_name_or_path=LOCAL_MODEL_PATH,
            device_mesh=actor_device_mesh,
        )

        # ----- Adapter first -----
        self.model.add_adapter_to_model(self.adapter_name, lora_config)

        # ----- Optimizer group MUST exist before set_loss(adapter_name=...) -----
        self.model.set_optimizer(
            "AdamW",
            lr=float(os.environ.get("TWINKLE_LR", "1e-6")),
            adapter_name=self.adapter_name,
        )
        self.model.set_lr_scheduler("LinearLR", adapter_name=self.adapter_name)

        # Now it is safe to set loss for this adapter
        self.model.set_loss(
            "GRPOLoss",
            adapter_name=self.adapter_name,
            loss_type="grpo",
            epsilon=float(os.environ.get("TWINKLE_EPSILON", "0.2")),
            beta=float(os.environ.get("TWINKLE_BETA", "0.04")),
            num_generations=int(os.environ.get("TWINKLE_NUM_GENERATIONS", "1")),
            scale_rewards="group",
        )

        self.model.set_processor("InputProcessor", adapter_name=self.adapter_name)
        self.model.set_template("Qwen3Template", adapter_name=self.adapter_name)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH,
            trust_remote_code=True,
        )

        self.loss_processor = GRPOLossProcessor()

        # one-time flag for debugging batch structure
        self._printed_batch_schema = False

    def _get_torch_model_and_device(self):
        torch_model = getattr(self.model, "model", None)
        if torch_model is None:
            raise RuntimeError("TransformersModel has no attribute `.model` (torch module).")
        device = next(torch_model.parameters()).device
        return torch_model, device

    def _normalize_prompts(self, batch):
        """
        Convert various batch structures into list[str] for tokenizer.
        Supported common shapes:
          - str
          - list[str]
          - dict with key: prompts / prompt / input / text / question
          - list[dict] where dict contains prompt/input/text/question
          - list[tuple] (take first element)
        Fallback: stringify items to keep the minimal loop running.
        """
        if batch is None:
            return []

        # dict batch
        if isinstance(batch, dict):
            if "prompts" in batch and isinstance(batch["prompts"], list):
                return [str(x) for x in batch["prompts"]]

            for k in ("prompt", "input", "text", "question"):
                if k in batch and batch[k] is not None:
                    v = batch[k]
                    if isinstance(v, str):
                        return [v]
                    if isinstance(v, list):
                        return [str(x) for x in v]
                    return [str(v)]

            # fallback
            return [str(batch)]

        # single string
        if isinstance(batch, str):
            return [batch]

        # list batch
        if isinstance(batch, list):
            if len(batch) == 0:
                return []
            # list[str]
            if all(isinstance(x, str) for x in batch):
                return batch
            # list[dict]
            if all(isinstance(x, dict) for x in batch):
                out = []
                for x in batch:
                    for k in ("prompt", "input", "text", "question"):
                        if k in x and x[k] is not None:
                            out.append(str(x[k]))
                            break
                    else:
                        out.append(str(x))
                return out

            # mixed / tuple
            out = []
            for x in batch:
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, tuple) and len(x) > 0:
                    out.append(str(x[0]))
                elif isinstance(x, dict):
                    for k in ("prompt", "input", "text", "question"):
                        if k in x and x[k] is not None:
                            out.append(str(x[k]))
                            break
                    else:
                        out.append(str(x))
                else:
                    out.append(str(x))
            return out

        # fallback
        return [str(batch)]

    @torch.inference_mode()
    def _sample_local(self, batch):
        # One-time structural debug to know what DataLoader yields inside Ray actor
        if not self._printed_batch_schema:
            self._printed_batch_schema = True
            try:
                keys = list(batch.keys()) if isinstance(batch, dict) else None
            except Exception:
                keys = None
            print(f"[sample_local] pid={os.getpid()} batch_type={type(batch)} batch_keys={keys} file={__file__}")

        prompts = self._normalize_prompts(batch)
        if len(prompts) == 0:
            raise ValueError(f"Empty prompts after normalization. batch_type={type(batch)} batch={batch}")

        # Optional: keep minimal loop stable by sampling just 1 prompt
        # prompts = prompts[:1]

        torch_model, device = self._get_torch_model_and_device()

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = torch_model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        n = min(len(prompts), len(texts))
        return [{"prompt": prompts[i], "response": texts[i]} for i in range(n)]

    def _build_train_inputs_from_batch(self, batch):
        # Build a minimal LM training batch for GRPO processor: tensors on model device
        prompts = self._normalize_prompts(batch)
        if not prompts:
            raise ValueError(f"Empty prompts for training inputs. batch_type={type(batch)}")
        torch_model, device = self._get_torch_model_and_device()
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        # Minimal causal LM labels
        enc["labels"] = enc["input_ids"].clone()
        return AttrDict(enc)

    @remote_function()
    def sample(self, batch):
        return self._sample_local(batch)

    @remote_function()
    def forward(self, inputs=None, **kwargs):
        # TransformersModel.forward in this Twinkle build is kw-only.
        if inputs is None:
            return self.model.forward(**kwargs)
        if isinstance(inputs, dict):
            return self.model.forward(**inputs, **kwargs)
        return self.model.forward(inputs=inputs, **kwargs)

    @remote_function()
    def forward_backward(self, inputs, trajectories, ref_logits, **kwargs):
        # Ensure adapter_name always present
        kwargs.setdefault("adapter_name", self.adapter_name)
        torch_model, device = self._get_torch_model_and_device()
        inputs = _to_attr_inputs(inputs, device=device)
        inputs = self._build_train_inputs_from_batch(inputs)
        inputs = self.loss_processor(inputs)

        # forward_backward is also kw-only compatible
        fb_kwargs = dict(kwargs)
        fb_kwargs["trajectories"] = trajectories
        if ref_logits is not None:
            fb_kwargs["ref_logits"] = ref_logits

        if isinstance(inputs, dict):
            fb_kwargs.update(inputs)
            fb_kwargs["inputs"] = inputs
        return self.model.forward_backward(**fb_kwargs)

        fb_kwargs["inputs"] = inputs
        return self.model.forward_backward(**fb_kwargs)

    @remote_function()
    def step(self):
        return self.model.step()

    @remote_function()
    def zero_grad(self):
        return self.model.zero_grad()

    @remote_function()
    def lr_step(self):
        return self.model.lr_step()


def train():
    dataset = create_dataset()
    try:
        dataloader = DataLoader(dataset, remote_group="actor", device_mesh=actor_device_mesh)
        _ = iter(dataloader)
    except Exception as e:
        print(f"[WARN] DataLoader/dataset init failed, fallback to dummy batch. err={type(e).__name__}: {e}")
        # minimal local fallback for 1-step closure
        dataloader = [
            [{"prompt": "Compute 1+1.", "answer": "2"}],
        ]

    adapter_name = "default"

    lora_config = LoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=int(os.environ.get("LORA_R", "8")),
        lora_alpha=int(os.environ.get("LORA_ALPHA", "16")),
        lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
        bias="none",
        task_type="CAUSAL_LM",
    )

    actor_group = ActorGroup.remote(
        lora_config=lora_config,
        remote_group="actor",
        adapter_name=adapter_name,
        max_new_tokens=int(os.environ.get("TWINKLE_MAX_NEW_TOKENS", "64")),
    )

    ref_model = TransformersModel(
        pretrained_model_name_or_path=LOCAL_MODEL_PATH,
        remote_group="ref",
        device_mesh=ref_device_mesh,
    )
    ref_model.set_processor("InputProcessor")
    ref_model.set_template("Qwen3Template")
    # Ref model only needs forward; provide a dummy torch Optimizer to satisfy _lazy_wrap_model assertions
    ref_model.set_optimizer("AdamW", lr=0.0)
    ref_model.set_lr_scheduler("LinearLR")

    reward = MathReward()

    print("Device placement:", get_device_placement())
    print("LOCAL_MODEL_PATH:", LOCAL_MODEL_PATH)

    max_steps = int(os.environ.get("TWINKLE_MAX_STEPS", "1"))
    step = 0

    for batch in dataloader:
        trajectories = ray.get(actor_group.sample.remote(batch))
        ref_logits = None  # Goal A minimal loop: skip ref model to avoid OOM
        trajectories_for_reward = _wrap_trajectories_for_reward(trajectories)
        ground_truths_for_reward = _wrap_trajectories_for_reward(batch)
        trajectories = reward.calculate(trajectories_for_reward, ground_truths_for_reward)

        ray.get(actor_group.forward_backward.remote(batch, trajectories, ref_logits, adapter_name=adapter_name))
        actor_group.step()
        actor_group.zero_grad()
        actor_group.lr_step()

        step += 1
        if step >= max_steps:
            break


def main():
    os.environ.setdefault("TWINKLE_SEED", "42")
    os.environ.setdefault("TWINKLE_FULL_DETERMINISM", "0")
    os.environ.setdefault("TRUST_REMOTE_CODE", "1")

    twinkle.initialize(
        mode="ray",
        nproc_per_node=2,
        groups=device_groups,
        lazy_collect=False,
    )

    train()


if __name__ == "__main__":
    main()
