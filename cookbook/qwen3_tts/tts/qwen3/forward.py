"""Qwen3-TTS two-stage forward. Talker 使用内部时间 shift；残差码本 logits 交自定义同位 CE。
小型替身可测试调用契约；真实模型/NPU/LoRA 仍需单独验收。"""

from typing import Dict, Optional
import torch


def qwen3_tts_forward_step(
    model,
    batch: Dict[str, torch.Tensor],
    target_speaker_embedding: Optional[torch.Tensor] = None,
):
    """Qwen3-TTS 训练一步前向（产出中间态 dict，供 Qwen3TTSLoss 算总 loss）。

    Args:
        model: `Qwen3TTSForConditionalGeneration`（底层，= `Qwen3TTSModel.model`），需含顶层
               `.speaker_encoder` + `.talker`（含 `.model.codec_embedding`、`.model.text_embedding`、
               `.text_projection`、`.codec_head`、`.code_predictor`、`.forward_sub_talker_finetune`）。
        batch: collate_qwen3_batch 输出转 torch 后的张量字典（input_ids[B,T,2]、codec_ids[B,T,16]、
               text_embedding_mask[B,T,1]、codec_embedding_mask[B,T,1]、codec_mask[B,T]、
               attention_mask[B,T]、codec_0_labels[B,T]、ref_mels[B,F,128]）。
        target_speaker_embedding: 首步算出的目标说话人 emb（官方写进 codec_embedding.weight[3000]；
               本项目走 clone 推理，可传 None，仅用于一致性记录）。

    Returns:
        dict:
          - 'talker_loss': 码本0 的 mean CE（model.talker.forward 的 outputs.loss，标准时间轴 shift 正确）。
          - 'sub_talker_logits': code_predictor 输出 [N, 15, V]（N=批次目标帧总数；15=码本1-15 头数）。
          - 'sub_talker_labels': [N, 15] = talker_codec_ids[:, 1:]（码本1-15 真值，已是 head 预测目标序列）。
          - 'target_speaker_embedding': 首步缓存的 speaker emb（供外部记录）。
        总 loss = talker_loss + 0.3 × code_predictor CE，由 Qwen3TTSLoss 从本 dict 算（不在编排里算）。
    """
    talker = model.talker
    # ---- ① speaker_encoder 前置（独立前向，detach 冻结）----
    with torch.no_grad():
        speaker_embedding = model.speaker_encoder(batch['ref_mels'].to(model.dtype))
    if target_speaker_embedding is None:
        target_speaker_embedding = speaker_embedding

    # ---- ② 手动组装 inputs_embeds（不走 model(input_ids=...)）----
    input_text_ids = batch['input_ids'][..., 0]        # [B,T]
    input_codec_ids = batch['input_ids'][..., 1]      # [B,T]
    text_emb_mask = batch['text_embedding_mask'].to(model.dtype)
    codec_emb_mask = batch['codec_embedding_mask'].to(model.dtype)
    codec_ids = batch['codec_ids']                    # [B,T,16]
    codec_mask = batch['codec_mask']                  # [B,T]

    # 0.6B: text_hidden_size=2048, hidden_size=1024. Even equal-sized
    # variants must use the pretrained projection, as official inference does.
    input_text_emb = talker.text_projection(talker.model.text_embedding(input_text_ids)) * text_emb_mask
    input_codec_emb = talker.model.codec_embedding(input_codec_ids) * codec_emb_mask
    input_codec_emb[:, 6, :] = speaker_embedding      # 注入 pos6
    input_embeddings = input_text_emb + input_codec_emb

    # 叠加码本 1-15 的输入嵌入（code_predictor.get_input_embeddings() 返回 ModuleList）
    cp_input_embs = talker.code_predictor.get_input_embeddings()   # ModuleList[15]
    for i in range(1, talker.config.num_code_groups):
        e = cp_input_embs[i - 1](codec_ids[..., i]) * codec_mask.unsqueeze(-1).to(model.dtype)
        input_embeddings = input_embeddings + e

    # ---- ③ 完整时间序列交 Talker；内部 ForCausalLMLoss 仅 shift 一次。----
    outputs = talker(
        inputs_embeds=input_embeddings,
        attention_mask=batch['attention_mask'],
        labels=batch['codec_0_labels'],
        output_hidden_states=True,
        use_cache=False,
    )
    talker_loss = outputs.loss                            # 码本0 mean shift+CE（HF loss_function，时间轴正确）

    # ---- ④ 第二段 forward：Code predictor 预测码本 1-15（向量化，一次喂所有目标帧）----
    # 注：forward_sub_talker_finetune 已向量化（输入 [N,16]+[N,H] → 内部 [N,16,H] 一次前向），
    hidden = outputs.hidden_states[0][-1]                 # [B, T, H]
    talker_hidden = hidden[:, :-1][codec_mask[:, 1:]]                    # Use the state that predicted cb0, before consuming this frame; no target leakage
    talker_codec_ids = codec_ids[codec_mask]             # [N, 16]
    sub_talker_logits, _sub_loss_ignored = talker.forward_sub_talker_finetune(
        talker_codec_ids, talker_hidden
    )
    # 忽略 _sub_loss_ignored（继承的 ForCausalLMLoss 对码本轴有 off-by-one，见 loss.py）；
    # Qwen3TTSLoss 用 sub_talker_logits + sub_talker_labels 自己算正确的 per-head CE。
    sub_talker_labels = talker_codec_ids[:, 1:]          # [N, 15] = 码本1-15 真值

    return {
        'talker_loss': talker_loss,
        'sub_talker_logits': sub_talker_logits,          # [N, 15, V]
        'sub_talker_labels': sub_talker_labels,           # [N, 15]
        'target_speaker_embedding': target_speaker_embedding,
    }
