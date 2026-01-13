# 截止0.1版本release

发布时间: 1.31
联合FC上线时间: 春节前

## 基础能力

- [x] sft跑通
- [x] ray sft跑通
- [x] http sft跑通
- [ ] 多模态模型支持 P0 @jinghan
- [ ] megatron支持 P0 @jinghan
    - [ ] megatron多模态模型 P0
- [ ] kernels/modules支持 P0
    - [ ] liger kernel支持 P1
- [ ] padding_free支持 P0 @yuhong
- [ ] GRPO/多模态GRPO支持 P0 @yimin
  - [ ] model/sampler placement（同步权重） P0 @yimin
- [ ] DAPO、GSPO等算法平迁 P1
- [ ] GKD/on-policy-distill平迁 P1
- [ ] streaming数据支持 P0 @yuhong
- [ ] MFU调优 P0 @yuhong
- [ ] PT训练支持 P0 @yimin
- [ ] DPO训练支持 P1
- [ ] ulysses+ring-attention平迁 P1

## 多租户

- [ ] multi-lora支持 P0 @jinghan
    - [ ] transformers P0 @jinghan
    - [ ] megatron P0 @jinghan
- [ ] 对接FC、百炼，可对外 P0 @yuhong @yunlin
  - [ ] 多租户水位和降级控制 P0 @yuhong @yunlin

## API

- [ ] 优化twinkle_client P0 @yunlin
- [ ] 支持tinker client P0 @yunlin

## 文档和发布

- [ ] code doc补充 P0 @yuhong
- [ ] 测试用例补充 P0 @all
- [ ] 文档编写 P0 @yuhong
- [ ] README P0 @yuhong
- [ ] 文档站建立 P0 @yuhong