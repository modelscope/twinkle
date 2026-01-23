print("[cookbook] ENTRY OK", flush=True)
import ray
# cookbook/grpo/lora.py

import torch
from twinkle.data_format import Trajectory, Message

from twinkle.infra import DiagnosticsConfig
import os
import time
import logging
from typing import Dict, Any, Optional

def _setup_logger(name: str = "twinkle_grpo") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
    )
    h = logging.StreamHandler()
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger

LOGGER = _setup_logger()

def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _ray_cluster_snapshot() -> Dict[str, Any]:
    import ray
    ray.init(address="auto", ignore_reinit_error=True)
    return {
        "cluster_resources": ray.cluster_resources(),
        "available_resources": ray.available_resources(),
    }

def _ray_min_gpu_probe(timeout_s: int = 60) -> Dict[str, Any]:
    """
    Submit a minimal GPU task to prove:
    - Ray sees GPU resources
    - a GPU worker can actually start
    This isolates infrastructure/resource issues from algorithm issues.
    """
    import ray
    ray.init(address="auto", ignore_reinit_error=True)

    @ray.remote(num_gpus=1)
    def _probe():
        import os
        import torch
        out = {
            "pid": os.getpid(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            out["cuda_name0"] = torch.cuda.get_device_name(0)
        return out

    ref = _probe.remote()
    # wait with timeout to avoid hanging forever
    start = time.time()
    while True:
        ready, _ = ray.wait([ref], timeout=1.0)
        if ready:
            return ray.get(ready[0])
        if time.time() - start > timeout_s:
            raise TimeoutError(f"GPU probe timed out after {timeout_s}s (likely resource starvation).")

def _is_resource_starvation(snapshot: Dict[str, Any], need_gpu: float = 1.0, need_cpu: float = 1.0) -> bool:
    """
    Decide if current state is clearly resource starvation:
    - cluster has GPUs but available GPUs < need_gpu
    - or cluster GPUs == 0 (Ray didn't see GPUs)
    """
    cr = snapshot.get("cluster_resources", {}) or {}
    ar = snapshot.get("available_resources", {}) or {}
    cluster_gpu = float(cr.get("GPU", 0.0))
    avail_gpu = float(ar.get("GPU", 0.0))
    avail_cpu = float(ar.get("CPU", 0.0))

    if cluster_gpu <= 0.0:
        return True
    if avail_gpu < need_gpu:
        return True
    if avail_cpu < need_cpu:
        # CPU starvation is rarer but still blocks scheduling
        return True
    return False

def preflight_resource_check(
    *,
    need_gpu: float = 1.0,
    need_cpu: float = 1.0,
    do_gpu_probe: bool = True,
    probe_timeout_s: int = 90,
) -> None:
    """
    Fail-fast with a clear diagnostic message if the job cannot run due to infra/resources.
    This prevents misattributing scheduling issues to algorithm/logic issues.
    Controlled by env:
      - TWINKLE_PREFLIGHT=1 (default) enable
      - TWINKLE_GPU_PROBE=1 enable minimal GPU probe (default 1)
    """
    if os.environ.get("TWINKLE_PREFLIGHT", "1") != "1":
        LOGGER.info("[preflight] disabled by TWINKLE_PREFLIGHT!=1")
        return

    LOGGER.info("[preflight] checking Ray cluster resources ...")
    snap = _ray_cluster_snapshot()
    LOGGER.info("[preflight] cluster_resources=%s", snap["cluster_resources"])
    LOGGER.info("[preflight] available_resources=%s", snap["available_resources"])

    if _is_resource_starvation(snap, need_gpu=need_gpu, need_cpu=need_cpu):
        # This is *not* algorithm logic. It's infra/resources.
        raise RuntimeError(
            "Ray resource starvation or GPU not visible to Ray. "
            f"Need GPU>={need_gpu}, CPU>={need_cpu}. "
            f"cluster={snap['cluster_resources']} available={snap['available_resources']}. "
            "This indicates resource/placement issues (e.g., GPU=0, or GPUs held by other actors), "
            "not training algorithm logic."
        )

    if do_gpu_probe and os.environ.get("TWINKLE_GPU_PROBE", "1") == "1":
        LOGGER.info("[preflight] running minimal GPU probe task ...")
        info = _ray_min_gpu_probe(timeout_s=probe_timeout_s)
        LOGGER.info("[preflight] GPU probe ok: %s", info)

def post_submit_watchdog(
    *,
    start_ts: float,
    warn_after_s: int = 60,
    hard_fail_after_s: int = 300,
) -> None:
    """
    Optional watchdog: if your program is stuck waiting for scheduling,
    emit a strong hint before users suspect 'logic bug'.
    Controlled by TWINKLE_WATCHDOG=1 (default 1)
    """
    if os.environ.get("TWINKLE_WATCHDOG", "1") != "1":
        return
    elapsed = int(time.time() - start_ts)
    if elapsed >= warn_after_s:
        try:
            snap = _ray_cluster_snapshot()
            if _is_resource_starvation(snap, need_gpu=1.0, need_cpu=1.0):
                LOGGER.warning(
                    "[watchdog] still waiting after %ss; likely scheduling/resource starvation. "
                    "cluster=%s available=%s",
                    elapsed, snap["cluster_resources"], snap["available_resources"]
                )
                if elapsed >= hard_fail_after_s:
                    raise RuntimeError(
                        f"Hard-fail after {elapsed}s waiting; this is almost certainly resource starvation, "
                        "not algorithm logic. Please free GPUs / restart ray / reduce actors."
                    )
        except Exception as e:
            LOGGER.warning("[watchdog] snapshot failed: %s: %s", type(e).__name__, e, exc_info=True)

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
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
ngpu = len([x for x in visible.split(",") if x.strip() != ""]) if visible else torch.cuda.device_count()
ngpu = max(1, ngpu)
if ngpu >= 2:
    device_groups = [
        DeviceGroup(name="actor", ranks=[0], device_type="GPU"),
        DeviceGroup(name="ref",   ranks=[1], device_type="GPU"),
    ]
else:
    device_groups = [
        DeviceGroup(name="actor", ranks=[0], device_type="GPU"),
    ]

actor_device_mesh = DeviceMesh(device_type="cuda", mesh=[0], mesh_dim_names=("dp",))
ref_device_mesh = None
if ngpu >= 2:
    ref_device_mesh = DeviceMesh(device_type="cuda", mesh=[1], mesh_dim_names=("dp",))



def create_dataset():
    if os.environ.get("TWINKLE_SKIP_DATASET", "0") == "1":
        raise RuntimeError("Skip dataset by TWINKLE_SKIP_DATASET=1")

    dataset = Dataset(DatasetMeta("ms://modelscope/competition_math"))
    try:
        dataset.set_template("Qwen3Template", model_id=LOCAL_MODEL_PATH)
    except Exception as e:
        print(f"[WARN] set_template failed, continue without template: {type(e).__name__}: {e}")
    dataset.map("CompetitionMathProcessor")
    dataset.check(batched=True)
    return dataset
@remote_class()
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
    # 1) dataset/dataloader
    dataloader = None
    try:
        dataset = create_dataset()
    except Exception as e:
        print(f"[WARN] create_dataset failed, fallback to dummy batch. err={type(e).__name__}: {e}")
        dataset = None

    if dataset is not None:
        try:
            dataloader = DataLoader(dataset, remote_group="actor", device_mesh=actor_device_mesh)
            _ = iter(dataloader)
        except Exception as e:
            print(f"[WARN] DataLoader/dataset init failed, fallback to dummy batch. err={type(e).__name__}: {e}")
            dataloader = None

    if dataloader is None:
        dataloader = [
            [{"prompt": "Compute 1+1.", "answer": "2"}],
        ]

    # 2) actor lora
    adapter_name = "default"
    lora_config = LoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=int(os.environ.get("LORA_R", "8")),
        lora_alpha=int(os.environ.get("LORA_ALPHA", "16")),
        lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
        bias="none",
        task_type="CAUSAL_LM",
    )

    actor_group = ActorGroup(
        lora_config=lora_config,
        remote_group="actor",   # 允许传；__init__ 有 **kwargs
        adapter_name=adapter_name,
        max_new_tokens=int(os.environ.get("TWINKLE_MAX_NEW_TOKENS", "64")),
    )

    # 3) ref is optional (but if single GPU, you should also remove ref from initialize/groups)
    use_ref = os.environ.get("TWINKLE_USE_REF", "0") == "1"
    ref_logits = None
    if use_ref:
        ref_model = TransformersModel(
            pretrained_model_name_or_path=LOCAL_MODEL_PATH,
            remote_group="ref",
            device_mesh=ref_device_mesh,
        )
        ref_model.set_processor("InputProcessor")
        ref_model.set_template("Qwen3Template")
        # ref forward not wired yet; keep ref_logits=None for now

    reward = MathReward()

    print("Device placement:", get_device_placement())
    print("LOCAL_MODEL_PATH:", LOCAL_MODEL_PATH)

    max_steps = int(os.environ.get("TWINKLE_MAX_STEPS", "1"))
    step = 0
    for batch in dataloader:
        # 1) sampling: ideally returns List[Trajectory] once you switch to VLLMSampler
        trajectories = ray.get(actor_group.sample.remote(batch))

        # 2) build ground truths as List[Trajectory] for MathReward
        from twinkle.data_format import Trajectory, Message
        ground_truths_for_reward = []
        if isinstance(batch, list) and batch and isinstance(batch[0], dict):
            for row in batch:
                gt = (
                    row.get("answer")
                    or row.get("ground_truth")
                    or row.get("label")
                    or row.get("output")
                    or ""
                )
                ground_truths_for_reward.append(
                    Trajectory(messages=[Message(role="assistant", content=str(gt))])
                )
        else:
            # fallback if dataloader yields non-list/dict
            ground_truths_for_reward = [
                Trajectory(messages=[Message(role="assistant", content=str(batch))])
            ]

        # 3) if your sampler still returns dicts, keep compatibility via wrapper
        trajectories_for_reward = _wrap_trajectories_for_reward(trajectories)

        # 4) reward.calculate returns List[float]; write back to trajectories (required by GRPOLoss)
        rewards = reward.calculate(trajectories_for_reward, ground_truths_for_reward)

        # IMPORTANT: the `trajectories` passed into GRPOLoss must have `.rewards`
        if trajectories and isinstance(trajectories[0], dict) and "messages" in trajectories[0]:
            for t, r in zip(trajectories, rewards):
                t["rewards"] = [float(r)]
            trajs_for_loss = trajectories
        else:
            for t, r in zip(trajectories_for_reward, rewards):
                t.rewards = [float(r)]
            trajs_for_loss = trajectories_for_reward

        ray.get(actor_group.forward_backward.remote(batch, trajs_for_loss, ref_logits, adapter_name=adapter_name))
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

    # 关键：启动前判定“是不是资源问题”
    preflight_resource_check(need_gpu=1.0, need_cpu=1.0, do_gpu_probe=True, probe_timeout_s=90)
    diag = DiagnosticsConfig(
        enabled=True,
        need_gpu=1.0,
        need_cpu=1.0,
        gpu_probe=True,          # 用一次最小 GPU remote probe 来验证调度链路
        probe_timeout_s=60,
        watchdog=True,           # 资源心跳
        watchdog_interval_s=30,
        warn_after_s=60,
        hard_fail_after_s=300,   # 目前只会 ERROR 日志，不会强杀进程
    )
    twinkle.initialize(
        mode="ray",
        nproc_per_node=ngpu if ngpu >= 1 else 1,
        groups=device_groups,
        lazy_collect=False,
        diagnostics=diag,
    )

    train()

if __name__ == "__main__":
    main()
