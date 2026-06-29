install_twinkle_with_kernels() {
    pip install ".[test,client,server]" -i https://mirrors.aliyun.com/pypi/simple/ || pip install ".[test,client,server]"
}

if [ "$MODELSCOPE_SDK_DEBUG" == "True" ]; then
    # pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    git config --global --add safe.directory /twinkle
    git config --global user.email tmp
    git config --global user.name tmp.com

    # linter test
    # use internal project for pre-commit due to the network problem
    if [ `git remote -v | grep alibaba  | wc -l` -gt 1 ]; then
        pre-commit run -c .pre-commit-config_local.yaml --all-files
        if [ $? -ne 0 ]; then
            echo "linter test failed, please run 'pre-commit run --all-files' to check"
            echo "From the repository folder"
            echo "Run 'pre-commit install' install pre-commit hooks."
            echo "Finally run linter with command: 'pre-commit run --all-files' to check."
            echo "Ensure there is no failure!!!!!!!!"
            exit -1
        fi
    fi

    pip install decord einops -U -i https://mirrors.aliyun.com/pypi/simple/
    pip uninstall autoawq -y
    pip uninstall lmdeploy -y
    pip uninstall tensorflow -y
    pip install ray==2.48
    pip install optimum

    # test with install
    install_twinkle_with_kernels
    # pyproject caps peft<=0.19.0, but 0.19.0 still does an unconditional
    # `from transformers import HybridCache` at peft_model.py:37 which
    # crashes on transformers v5. 0.19.1 dropped that top-level import.
    pip install --upgrade 'peft>=0.19.1'
    # Pin huggingface_hub AFTER main install to prevent transitive upgrade.
    # kernels<0.15 uses str | None (PEP 604) which newer huggingface_hub's
    # strict dataclass validator rejects (huggingface/transformers#46291).
    pip install 'huggingface_hub<0.31'
    pip install 'kernels<0.15'
else
    install_twinkle_with_kernels
    # Same kernels pin and peft bump for the release-image branch.
    pip install --upgrade 'peft>=0.19.1'
    # Pin huggingface_hub AFTER main install (same reason as debug branch).
    pip install 'huggingface_hub<0.31'
    pip install 'kernels<0.15'
    echo "Running case in release image, run case directly!"
fi
# remove torch_extensions folder to avoid ci hang.
rm -rf ~/.cache/torch_extensions
if [ $# -eq 0 ]; then
    ci_command="pytest tests"
else
    ci_command="$@"
fi
echo "Running case with command: $ci_command"
$ci_command
