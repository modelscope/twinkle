# ModelScope Free Resources

Alongside the open-sourcing of the Twinkle framework, we provide free RL training resources on the [ModelScope Community website](https://www.modelscope.cn). Developers only need to provide a ModelScope SDK token to train **for free**.

The models currently running on the cluster are: Below are the specific usage methods:

## Step 1. Register as a ModelScope User

Developers first need to register as ModelScope users and use the ModelScope community token for API calls.

Registration URL: https://www.modelscope.cn/

Get your token here: https://www.modelscope.cn/my/access/token Copy the access token to use in the SDK.

## Step 2. Join the twinkle-explorers Organization

Currently, twinkle-kit's remote training capability is in beta testing. Developers need to join the [twinkle-explorers](https://www.modelscope.cn/models/twinkle-explorers) organization. Users within this organization can participate in early usage and testing.
There is no threshold for applying to and joining this organization; it is currently only used for traffic control and bug feedback in the early stages of the project launch. After the project stabilizes, we will remove the organization join restriction.

## Step 3. Check the Cookbook and Customize Development

We strongly recommend that developers check our [cookbook](https://github.com/modelscope/twinkle/tree/main/cookbook/client) and perform secondary development based on the training code therein.

Developers can customize datasets, advantage functions, rewards, templates, etc. The Loss component currently does not support customization because it needs to be executed on the server side (for security reasons).
If you need to support additional Loss functions, you can upload the Loss implementation to ModelHub and contact us in the Q&A group or through an issue to whitelist the corresponding component for use.
