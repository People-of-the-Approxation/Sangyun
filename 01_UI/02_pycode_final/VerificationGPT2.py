import torch
import numpy as np
import serial
import time
from typing import Optional, Tuple
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from softmax_batch import open_serial, close_serial, softmax_batch


SERIAL_PORT = "COM3"
BAUD_RATE = 115200


class GPT2AttentionSoftmaxApprox(GPT2Attention):

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)
        self.ser = None

    def set_serial(self, ser):
        self.ser = ser

    def _my_split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _my_merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if self.ser is None:
            raise RuntimeError("UART serial is not set. Call set_serial(ser).")

        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._my_split_heads(query, self.num_heads, self.head_dim)
        key = self._my_split_heads(key, self.num_heads, self.head_dim)
        value = self._my_split_heads(value, self.num_heads, self.head_dim)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]

        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(
            causal_mask.bool(),
            attn_weights,
            torch.tensor(mask_value).to(attn_weights.device),
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        B, H, Tq, Tk = attn_weights.shape
        attn_weights_cpu = attn_weights.detach().cpu().numpy()

        attn_probs = torch.zeros_like(attn_weights)

        for b in range(B):
            for h in range(H):
                matrix = attn_weights_cpu[b, h]
                rows_list = [matrix[i, :] for i in range(Tq)]

                probs_list = softmax_batch(
                    self.ser, rows_list, pad_value=-32.0, timeout_s=5.0
                )

                probs_matrix = np.vstack(probs_list)
                attn_probs[b, h] = torch.tensor(
                    probs_matrix, dtype=attn_weights.dtype, device=attn_weights.device
                )

        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, value)

        attn_output = self._my_merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        present = (key, value) if use_cache else None

        return attn_output, present


def replace_gpt2_attention(model: GPT2LMHeadModel, ser_instance):
    count = 0
    for i, layer in enumerate(model.transformer.h):
        old_attn = layer.attn
        new_attn = GPT2AttentionSoftmaxApprox(
            model.config,
            is_cross_attention=old_attn.is_cross_attention,
            layer_idx=old_attn.layer_idx,
        )
        new_attn.load_state_dict(old_attn.state_dict(), strict=True)
        new_attn.set_serial(ser_instance)

        layer.attn = new_attn
        count += 1
    print(f"Replaced {count} attention layers with Hardware-Approximated version.")


def build_model_GPT2(ser: serial.Serial):
    device = "cpu"
    model_name = "gpt2"

    print(f"Loading {model_name} model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    approx_model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    replace_gpt2_attention(approx_model, ser)
    return tokenizer, base_model, approx_model, device


def get_last_gpt2_attention_matrix(model, layer=0, head=0):
    """
    특정 레이어와 헤드에 저장된 마지막 Attention Map을 꺼내오는 함수입니다.
    """
    target_layer_idx = int(layer)
    target_head_idx = int(head)
    current_layer_idx = 0

    for name, module in model.named_modules():
        if isinstance(module, GPT2AttentionSoftmaxApprox):
            if current_layer_idx == target_layer_idx:
                if module.last_attn is not None:
                    try:
                        # last_attn shape: (Batch, Heads, T, T)
                        # 배치 0번의 특정 헤드 데이터를 가져옵니다.
                        return module.last_attn[0, target_head_idx, :, :].numpy()
                    except IndexError:
                        print(f"[Warning] Head index {target_head_idx} out of bounds.")
                        return None
                else:
                    return None
            current_layer_idx += 1

    print(f"[Warning] Layer {layer} not found.")
    return None
