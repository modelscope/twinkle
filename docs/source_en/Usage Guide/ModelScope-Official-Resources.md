# ModelScope Official Environment

Alongside the open-source release of the Twinkle framework, we provide sample training resources on the [ModelScope Community website](https://www.modelscope.cn). Developers can conduct training using their ModelScope Token.

The model currently running on the cluster is [Qwen/Qwen3-30B-A3B-Instruct-2507](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B-Instruct-2507). Below are the detailed usage instructions:

## Step 1. Register a ModelScope Account

Developers first need to register as a ModelScope user and use the ModelScope community token for API calls.

Registration URL: https://www.modelscope.cn/

The API endpoint is: https://www.modelscope.cn/twinkle

Token can be obtained here: https://www.modelscope.cn/my/access/token — copy your access token.

## Step 2. Join the twinkle-explorers Organization

Currently, the remote training capability of twinkle-kit is in beta testing. Developers need to join the [twinkle-explorers](https://www.modelscope.cn/models/twinkle-explorers) organization. Users within the organization can access and test these features.

## Step 3. Review the Cookbook and Customize Your Development

We strongly recommend that developers review our [cookbook](https://github.com/modelscope/twinkle/tree/main/cookbook/client) and build upon the training code provided there.

Developers can customize datasets, advantage functions, rewards, templates, and more. However, the Loss component is not currently customizable (for security reasons) since it needs to be executed on the server side.

If you need support for additional custom Loss functions, you can upload your Loss implementation to ModelHub and contact us through the Q&A group or via GitHub issues. We will add the corresponding component to the whitelist for your use.

## Appendix: Supported Training Methods

This model is a text-only model, so multimodal tasks are not supported at this time. For text-only tasks, you can train using:

1. Standard PT/SFT training methods, including Agentic training
2. Self-sampling RL algorithms such as GRPO/RLOO
3. Distillation methods like GKD/On-policy. Since the official ModelScope environment only supports a single model, developers need to prepare the other Teacher/Student model themselves

The current official environment only supports LoRA training, with the following requirements:

1. Maximum rank = 32
2. modules_to_save is not supported
