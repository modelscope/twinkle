# Contributor Guidelines

*Welcome to contribute Feature PRs, Bug reports, documentation, or other types of contributions to twinkle!*

## Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Contribution Process](#-contribution-process)
- [Resource Support](#-resource-support)

## 📖 Code of Conduct

Please refer to our [Code of Conduct document](./CODE_OF_CONDUCT.md).

## 🔁 Contribution Process

### What We Need

- New components: You can contribute excellent components to the twinkle project, or contribute them to the modelhub in the ModelScope/Hugging Face community following the component protocol, making them available for other developers to use
- New kernels: You can contribute low-level kernels to the twinkle project. These kernels can be integrated into models to achieve better training value

Your contributions will help other developers. Please add your component name, location, and usage documentation link in the Community Components section of the README in your code PR.

### Incentives

- We will issue electronic certificates to contributors on behalf of the ModelScope community to acknowledge your selfless contributions.
- We will give away ModelScope community merchandise and small gifts.

### Submitting PRs (Pull Requests)

All feature development is conducted on GitHub using a Fork-then-PR workflow.

1. Fork: Go to the [twinkle](https://github.com/modelscope/twinkle) page and click the **Fork button**. This will clone a twinkle repository under your personal organization

2. Clone: Clone the repository created in step one to your local machine and **create a new branch** for development. During development, please click the **Sync Fork button** regularly to sync with the `main` branch to prevent code from becoming outdated and causing conflicts

3. Submit PR: After development and testing are complete, push your code to the remote branch. On GitHub, click the **Pull Requests page** and create a new PR. Select your code branch as the source branch and `modelscope/twinkle:main` as the target branch

4. Write Description: It is essential to provide a good feature description in your PR so that reviewers understand your changes

5. Review: We want the merged code to be clean and efficient, so we may raise some questions for discussion. Please note that any questions raised during review are about the code itself, not about you personally. Once all issues have been discussed and resolved, your code will be approved

### Code Standards and Development Practices

twinkle has established conventions for variable naming and development practices. Please try to follow these conventions during development.

1. Variable names use underscore separation; class names use PascalCase (capitalize the first letter of each word)
2. All Python indentation uses four spaces instead of one tab
3. Use well-known open-source libraries; avoid closed-source or unstable open-source libraries; avoid reinventing the wheel

twinkle runs two types of tests after a PR is submitted:

- Code Lint Tests: Static code analysis tests. To ensure this test passes, please run Code lint locally beforehand. Here's how:

  ```shell
  pip install pre-commit
  pre-commit run --all-files
  # Fix any errors reported by pre-commit until all checks pass
  ```

- CI Tests: Smoke tests and unit tests. Please refer to the next section

### Running CI Tests

Before submitting a PR, please ensure your development code is protected by test cases. For example, smoke tests for new features, or unit tests for various edge cases. Reviewers will also pay attention to this during code review. Additionally, a dedicated service will run CI Tests, executing all test cases. Code can only be merged after all test cases pass.

Please ensure these tests pass successfully.
