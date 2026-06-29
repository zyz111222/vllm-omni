# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Qwen3-TTS for the 310P NPU path."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mimi import modeling_mimi
from vllm.logger import init_logger
from vllm.multimodal.audio import AudioResampler

from vllm_omni.model_executor.models.common import qwen3_code_predictor
from vllm_omni.model_executor.models.qwen3_tts import (
    prompt_embeds_builder,
    qwen3_tts_talker,
)

_RUNTIME_DTYPE = torch.float16
_CPU_DEVICE = torch.device("cpu")
_PATCHED = False
_ORIGINAL_CODE_PREDICTOR_SETUP_COMPILE = None
_ORIGINAL_CODE_PREDICTOR_FORWARD = None

logger = init_logger(__name__)


class _MimiEuclideanCodebook310P(modeling_mimi.MimiEuclideanCodebook):
    def quantize(self, hidden_states):
        # 310P does not support torch.cdist on NPU.
        device = hidden_states.device
        dists = torch.cdist(
            hidden_states[None].to(_CPU_DEVICE, torch.float32), self.embed[None].to(_CPU_DEVICE, torch.float32), p=2
        )[0]
        return dists.argmin(dim=-1).to(device=device)


class _Qwen3TTSTalker310P(qwen3_tts_talker.Qwen3TTSTalkerForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._embedding_dtype = _RUNTIME_DTYPE
        self._prompt_builder._embedding_dtype = _RUNTIME_DTYPE

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        self.encoder.to(dtype=_RUNTIME_DTYPE)
        return loaded

    def _encode_ref_audio_batch(
        self,
        wavs: list[np.ndarray],
        sr: int,
        *,
        device: torch.device,
    ) -> list[torch.Tensor]:
        fe = self._encoder_feature_extractor
        target_sr = int(fe.sampling_rate)
        if int(sr) != target_sr:
            resampler = AudioResampler(target_sr=target_sr)
            wavs = [resampler.resample(w.astype(np.float32), orig_sr=int(sr)) for w in wavs]

        inputs = fe(raw_audio=wavs, sampling_rate=target_sr, return_tensors="pt").to(device).to(_RUNTIME_DTYPE)

        with torch.inference_mode():
            encoded = self.encoder.encode(
                input_values=inputs["input_values"].squeeze(1).unsqueeze(1),
                return_dict=True,
            )

        audio_codes = encoded.audio_codes[:, : self._encoder_valid_num_quantizers]
        padding_mask = inputs["padding_mask"].squeeze(1)
        downsample = self._encoder_downsample_rate
        return [
            code[..., : -(-mask.sum() // downsample)].transpose(0, 1).to(device=device, dtype=torch.long)
            for code, mask in zip(audio_codes, padding_mask)
        ]


class _Qwen3TTSPromptEmbedsBuilder310P(prompt_embeds_builder.Qwen3TTSPromptEmbedsBuilder):
    def extract_speaker_embedding(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        dev = self._device()
        dtype = self._embedding_dtype
        try:
            spk_param = next(self._speaker_encoder.parameters())
            if spk_param.device != dev or spk_param.dtype != dtype:
                self._speaker_encoder.to(device=dev, dtype=dtype)
        except StopIteration:
            pass

        target_sr = int(getattr(self._config.speaker_encoder_config, "sample_rate", 24000))
        if sr != target_sr:
            resampler = self._get_resampler(int(sr), target_sr)
            wav = resampler.resample(wav.astype(np.float32), orig_sr=int(sr))

        # 310P does not support torch.stft on NPU.
        wav_tensor = torch.from_numpy(wav).to(device=_CPU_DEVICE, dtype=torch.float32).unsqueeze(0)
        mels = prompt_embeds_builder.mel_spectrogram(
            wav_tensor,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        spk = self._speaker_encoder(mels.to(device=dev, dtype=dtype))[0]
        return spk.to(dtype=dtype)


class _RMSNorm310P(nn.Module):
    """310P RMSNorm implementation backed by the Ascend NPU fused operator."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type != "npu":
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)

        import torch_npu

        hidden_states, _ = torch_npu.npu_rms_norm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
        )
        return hidden_states


class _RotaryEmbedding310P(nn.Module):
    """RoPE cache for the short 310P CodePredictor re-prefill sequence.

    The generic implementation follows HuggingFace numerics and builds
    cos/sin every forward.  On 310P the sequence length is bounded by the
    codebook count, so the table is static and can be materialized once before
    NPU graph capture.
    """

    def __init__(self, config) -> None:
        super().__init__()
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        rope_theta = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        max_seq = int(getattr(config, "num_code_groups", 0) or 0) + 1
        positions = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


class _Qwen3CodePredictorAttention310P(qwen3_code_predictor.CodePredictorAttention):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._buffers.pop("_fusion_causal_mask", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.device.type != "npu" or attention_mask is None:
            return super().forward(hidden_states, position_embeddings, attention_mask=attention_mask)

        bsz, seq_len, _ = hidden_states.shape
        hidden_shape_q = (bsz, seq_len, self.num_heads, self.head_dim)
        hidden_shape_kv = (bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape_q)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape_kv)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape_kv).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        import torch_npu

        q = torch_npu.npu_rotary_mul(q, cos, sin)
        k = torch_npu.npu_rotary_mul(k, cos, sin)

        real_tokens = int(bsz) * int(seq_len)
        output_dtype = q.dtype

        from vllm_ascend.utils import aligned_16

        q_f = aligned_16(q.to(torch.float16).transpose(1, 2).reshape(real_tokens, self.num_heads, self.head_dim))
        k_f = aligned_16(k.to(torch.float16).transpose(1, 2).reshape(real_tokens, self.num_kv_heads, self.head_dim))
        v_f = aligned_16(v.to(torch.float16).transpose(1, 2).reshape(real_tokens, self.num_kv_heads, self.head_dim))

        aligned_tokens = int(q_f.shape[0])
        seq_lens = torch.full((int(bsz),), int(seq_len), dtype=torch.int32, device="cpu")
        if aligned_tokens > real_tokens:
            seq_lens[-1] += aligned_tokens - real_tokens

        out = torch.empty((aligned_tokens, self.num_heads, self.head_dim), dtype=torch.float16, device=q.device)
        torch_npu._npu_flash_attention(
            query=q_f.contiguous(),
            key=k_f.contiguous(),
            value=v_f.contiguous(),
            mask=attention_mask,
            seq_len=seq_lens,
            scale_value=float(self.scaling),
            num_heads=int(self.num_heads),
            num_kv_heads=int(self.num_kv_heads),
            out=out,
        )
        attn_out = out[:real_tokens].reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        return self.o_proj(attn_out.to(output_dtype).transpose(1, 2).reshape(bsz, seq_len, -1))


class _Qwen3CodePredictorDecoderLayer310P(qwen3_code_predictor.CodePredictorDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask=attention_mask)
        import torch_npu

        # Fuse the residual add and post-attention RMSNorm.  310P has an
        # efficient NPU kernel for this pattern, while the generic PyTorch
        # expression expands into multiple elementwise kernels.
        hidden_states, _, residual = torch_npu.npu_add_rms_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
        )
        return residual + self.mlp(hidden_states)


class _Qwen3CodePredictorBaseModel310P(qwen3_code_predictor.CodePredictorBaseModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._attention_mask_310p = None
        self._attention_mask_310p_device = None
        self._attention_mask_310p_max_seq = ((int(self.config.num_code_groups) + 16) // 16) * 16

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if inputs_embeds.device.type != "npu":
            return super().forward(inputs_embeds, position_ids)

        if self._attention_mask_310p is None or self._attention_mask_310p_device != inputs_embeds.device:
            import torch_npu
            from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_2d

            mask = AttentionMaskBuilder310.gen_causal_additive_mask(
                self._attention_mask_310p_max_seq,
                inputs_embeds.device,
            )
            self._attention_mask_310p = torch_npu.npu_format_cast(nd_to_nz_2d(mask), ACL_FORMAT_FRACTAL_NZ)
            self._attention_mask_310p_device = inputs_embeds.device

        input_dtype = inputs_embeds.dtype
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                attention_mask=self._attention_mask_310p,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states.to(input_dtype)


def _maybe_transform_linears_for_310p(module: nn.Module, *, log_prefix: str, attr_name: str) -> None:
    """Persistently convert Linear weights to the 310P matmul-friendly layout."""
    if getattr(module, attr_name, False):
        return

    from vllm_ascend.utils import maybe_trans_nz

    transformed = 0
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, nn.Linear):
                child.weight.data = maybe_trans_nz(child.weight.data)
                transformed += 1
    setattr(module, attr_name, True)
    if transformed:
        logger.info_once("%s: transformed %d Linear weights for 310P layout", log_prefix, transformed)


def _build_projected_codec_embed_weight_310p(self) -> None:
    if not hasattr(self, "_projected_codec_embed_weight"):
        self.register_buffer("_projected_codec_embed_weight", None, persistent=False)

    codec_embeds = self._codec_embeds_list
    if codec_embeds is None or self._wrapper_config.use_parallel_embedding:
        self._projected_codec_embed_weight = None
        return

    projected_weights: list[torch.Tensor] = []
    with torch.no_grad():
        for embed_layer in codec_embeds:
            if not isinstance(embed_layer, nn.Embedding):
                self._projected_codec_embed_weight = None
                return
            projected_weights.append(self.small_to_mtp_projection(embed_layer.weight).detach())
    self._projected_codec_embed_weight = torch.stack(projected_weights, dim=0).contiguous()


def _setup_compile_310p(self) -> None:
    if self._compiled_model_fwd is not None:
        return

    self._model_dtype = next(self.model.parameters()).dtype
    self.model.rotary_emb.to(device=next(self.model.parameters()).device, dtype=self._model_dtype)
    self._lm_heads_list = list(self.lm_head)
    self._codec_embeds_list = list(self.model.codec_embedding)

    _maybe_transform_linears_for_310p(
        self,
        log_prefix="code_predictor",
        attr_name="_linear_weights_transformed_for_310p",
    )
    _build_projected_codec_embed_weight_310p(self)

    if not qwen3_code_predictor.current_omni_platform.supports_torch_inductor():
        self._compiled_model_fwd = self.model.forward
        if qwen3_code_predictor.current_omni_platform.is_npu() and self._wrapper_config.use_cuda_graphs:
            self._warmup_buckets()
            self._capture_npu_graphs()
            logger.info("code_predictor: eager mode + NPU graphs")
        else:
            logger.warning_once("code_predictor: torch.compile disabled")
        return

    self._compiled_model_fwd = torch.compile(
        self.model.forward,
        dynamic=False,
        options={"epilogue_fusion": False},
    )
    self._warmup_buckets()

    if self._wrapper_config.use_cuda_graphs:
        self._capture_cuda_graphs()
        logger.info("code_predictor: torch.compile (no epilogue fusion) + CUDA graphs")
    else:
        logger.info("code_predictor: torch.compile (dynamic=False, no epilogue fusion)")


@torch.inference_mode()
def _code_predictor_forward_310p(
    self,
    layer0_code: torch.Tensor,
    layer0_embed: torch.Tensor,
    last_talker_hidden: torch.Tensor,
    do_sample: bool = True,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if layer0_code.device.type != "npu":
        assert _ORIGINAL_CODE_PREDICTOR_FORWARD is not None
        return _ORIGINAL_CODE_PREDICTOR_FORWARD(
            self,
            layer0_code,
            layer0_embed,
            last_talker_hidden,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )

    bsz = int(layer0_code.shape[0])
    num_groups = self._num_groups
    device = layer0_code.device

    self._setup_compile()
    dtype = self._model_dtype

    padded_bsz = self._padded_bsz(bsz)
    self._ensure_buffers(device, dtype, padded_bsz)

    proj_buf = self._proj_buf
    max_seq = num_groups + 1
    projection = self.small_to_mtp_projection
    model_fwd = self._compiled_model_fwd
    lm_heads = self._lm_heads_list
    codec_embeds = self._codec_embeds_list
    projected_codec_embed_weight = getattr(self, "_projected_codec_embed_weight", None)

    proj_buf[:padded_bsz].zero_()
    proj_buf[:bsz, 0, :] = projection(last_talker_hidden.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)
    proj_buf[:bsz, 1, :] = projection(layer0_embed.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)

    stored_mode = self._wrapper_config.sampling_mode == "stored"
    if stored_mode:
        s_top_k = self._top_k
        s_top_p = self._top_p
    else:
        use_sampling = do_sample and temperature > 0
        inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0
        if use_sampling and top_p != 1.0:
            raise NotImplementedError(
                "top_p sampling is not implemented for the vLLM-native code predictor; please set top_p=1.0."
            )

    if self._wrapper_config.return_proj_buf:
        all_codes = torch.empty(bsz, num_groups, 1, dtype=torch.int64, device=device)
        all_codes[:, 0] = layer0_code.reshape(bsz, -1)[:, :1]
    else:
        all_codes = torch.empty(bsz, num_groups, dtype=torch.long, device=device)
        all_codes[:, 0] = layer0_code.reshape(bsz)

    for step in range(1, num_groups):
        graph_key: int | tuple[int, int] = padded_bsz
        seq_len = max_seq
        if self._prefix_graphs_enabled:
            prefix_key = (padded_bsz, step + 1)
            if prefix_key in self._device_graphs:
                graph_key = prefix_key
                seq_len = step + 1
        pos_ids = self._bucket_pos_ids.get(graph_key)
        if pos_ids is None:
            pos_ids = (
                torch.arange(seq_len, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(padded_bsz, -1)
                .contiguous()
            )

        device_graph_entry = self._device_graphs.get(graph_key)
        if device_graph_entry is not None:
            device_graph_entry[0].replay()
            hidden_out = device_graph_entry[1]
        else:
            hidden_out = model_fwd(proj_buf[:padded_bsz, :seq_len, :], pos_ids)

        logits = lm_heads[step - 1](hidden_out[:bsz, step, :])

        if stored_mode:
            if s_top_k > 0:
                topk_vals, _ = logits.topk(s_top_k, dim=-1)
                logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
            if s_top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1, dtype=torch.float32)
                cumulative_probs = sorted_probs.cumsum(dim=-1)
                remove_mask = (cumulative_probs - sorted_probs) >= s_top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1, dtype=torch.float32)
            code = torch.multinomial(probs, num_samples=1, generator=generator)
        else:
            if use_sampling:
                scaled = logits * inv_temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                probs = F.softmax(scaled, dim=-1, dtype=torch.float32)
                code = torch.multinomial(probs, num_samples=1, generator=generator)
            else:
                code = logits.argmax(dim=-1, keepdim=True)

        if self._wrapper_config.return_proj_buf:
            all_codes[:, step] = code
        else:
            all_codes[:, step] = code.reshape(bsz)

        if step < num_groups - 1 or self._wrapper_config.return_proj_buf:
            if projected_codec_embed_weight is not None:
                proj_buf[:bsz, step + 1, :].copy_(
                    F.embedding(code.reshape(-1), projected_codec_embed_weight[step - 1])
                )
            else:
                new_embed = codec_embeds[step - 1](code)
                proj_buf[:bsz, step + 1, :].copy_(
                    projection(new_embed.reshape(bsz, 1, -1)).reshape(bsz, -1)
                )

    if self._wrapper_config.return_proj_buf:
        return all_codes, proj_buf[:bsz].clone()
    return all_codes


def apply_talker_patches() -> None:
    global _PATCHED
    global _ORIGINAL_CODE_PREDICTOR_SETUP_COMPILE
    global _ORIGINAL_CODE_PREDICTOR_FORWARD

    if _PATCHED:
        return

    modeling_mimi.MimiEuclideanCodebook = _MimiEuclideanCodebook310P
    qwen3_tts_talker.Qwen3TTSTalkerForConditionalGeneration = _Qwen3TTSTalker310P
    qwen3_tts_talker.Qwen3TTSPromptEmbedsBuilder = _Qwen3TTSPromptEmbedsBuilder310P
    prompt_embeds_builder.Qwen3TTSPromptEmbedsBuilder = _Qwen3TTSPromptEmbedsBuilder310P
    qwen3_code_predictor._RMSNorm = _RMSNorm310P
    qwen3_code_predictor._RotaryEmbedding = _RotaryEmbedding310P
    qwen3_code_predictor.CodePredictorAttention = _Qwen3CodePredictorAttention310P
    qwen3_code_predictor.CodePredictorDecoderLayer = _Qwen3CodePredictorDecoderLayer310P
    qwen3_code_predictor.CodePredictorBaseModel = _Qwen3CodePredictorBaseModel310P

    _ORIGINAL_CODE_PREDICTOR_SETUP_COMPILE = qwen3_code_predictor.CodePredictorWrapper._setup_compile
    _ORIGINAL_CODE_PREDICTOR_FORWARD = qwen3_code_predictor.CodePredictorWrapper.forward
    qwen3_code_predictor.CodePredictorWrapper._setup_compile = _setup_compile_310p
    qwen3_code_predictor.CodePredictorWrapper.forward = _code_predictor_forward_310p

    _PATCHED = True
