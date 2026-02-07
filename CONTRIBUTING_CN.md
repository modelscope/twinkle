# 贡献者指引

*欢迎帮twinkle提供Feature PR、Bug反馈、文档补充或其他类型的贡献！*

## 目录

- [代码规约](#-代码规约)
- [贡献流程](#-贡献流程)
- [资源支持](#-资源支持)

## 📖 代码规约

请查看我们的[代码规约文档](./CODE_OF_CONDUCT.md).

## 🔁 贡献流程

### 我们需要什么

- 新组件：您可以将优秀的组件贡献进twinkle项目，或按照组件协议贡献进ModelScope/Hugging Face社区的modelhub中，方便其他开发者使用
- 新kernels：您可以将底层kernels贡献进twinkle项目中，这些kernels可以被模型集成，实现更好的训练价值

您的贡献会帮助到其他开发者，请在代码PR中在README的社区组件章节中增加您的组件名称、位置和使用方法文档链接。

### 激励

- 我们会以魔搭社区的身份给贡献者颁发电子证书，以鼓励您的无私贡献。
- 我们会赠送相关魔搭社区相关周边小礼品。

### 提交PR（Pull Requests）

任何feature开发都在github上以先Fork后PR的形式进行。

1. Fork：进入[twinkle](https://github.com/modelscope/twinkle)页面后，点击**Fork按钮**执行。完成后会在您的个人组织下克隆出一个twinkle代码库

2. Clone：将第一步产生的代码库clone到本地并**拉新分支**进行开发，开发中请及时点击**Sync Fork按钮**同步`main`分支，防止代码过期并冲突

3. 提交PR：开发、测试完成后将代码推送到远程分支。在github上点击**Pull Requests页面**，新建一个PR，源分支选择您提交的代码分支，目标分支选择`modelscope/twinkle:main`分支

4. 撰写描述：在PR中填写良好的feature描述是必要的，让Reviewers知道您的修改内容

5. Review：我们希望合入的代码简洁高效，因此可能会提出一些问题并讨论。请注意，任何review中提出的问题是针对代码本身，而非您个人。在所有问题讨论通过后，您的代码会被通过

### 代码规范和开发方式

twinkle有约定俗成的变量命名方式和开发方式。在开发中请尽量遵循这些方式。

1. 变量命名以下划线分割，类名以所有单词首字母大写方式命名
2. 所有的python缩进都是四个空格取代一个tab
3. 选用知名的开源库，避免使用闭源库或不稳定的开源库，避免重复造轮子

twinkle在PR提交后会进行两类测试：

- Code Lint测试 对代码进行静态规范走查的测试，为保证改测试通过，请保证本地预先进行了Code lint。方法是：

  ```shell
  pip install pre-commit
  pre-commit run --all-files
  # 对pre-commit报的错误进行修改，直到所有的检查都是成功状态
  ```

- CI Tests 冒烟测试和单元测试，请查看下一章节

### Running CI Tests

在提交PR前，请保证您的开发代码已经受到了测试用例的保护。例如，对新功能的冒烟测试，或者各种边缘case的单元测试等。在代码review时Reviewers也会关注这一点。同时，也会有服务专门运行CI Tests，运行所有的测试用例，测试用例通过后代码才可以合并。

请保证该测试可以正常通过。
