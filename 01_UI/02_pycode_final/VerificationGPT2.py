import torch
import torch.nn as nn
import numpy as np
import copy
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

# [수정] 없는 클래스 import를 제거하고, softmax_batch 함수를 직접 가져옵니다.
from softmax_batch import softmax_batch


class GPT2AttentionSoftmaxApprox(GPT2Attention):
    """
    GPT2Attention의 _attn 메서드를 오버라이딩하여,
    Softmax 연산을 하드웨어(시리얼)를 통해 수행하고
    그 결과(Heatmap)를 저장하는 클래스입니다.
    """

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(
            config, is_cross_attention=is_cross_attention, layer_idx=layer_idx
        )
        self.config = config
        self.ser = getattr(config, "ser", None)

        # 시각화를 위해 마지막 Attention Weights를 저장할 변수
        self.last_attn = None

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 1. Q * K^T 연산 (기존 GPT-2 로직)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if self.is_cross_attention and self.layer_idx is None:
            raise ValueError("Layer index must be set for cross-attention")

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # ============================================================
        # [핵심 수정] Softmax 교체 로직
        # ============================================================

        # 하드웨어 근사 모드가 켜져 있고, 시리얼 포트가 연결되어 있을 때
        if getattr(self.config, "use_approx", False) and self.ser is not None:
            # attn_weights shape: (Batch, Num_Heads, Seq_Len, Seq_Len)
            B, H, T, _ = attn_weights.shape

            # 1) 텐서를 Numpy(CPU)로 변환 및 펼치기
            # 시리얼 통신 효율을 위해 (Batch * Heads * Seq_Len)개의 행 벡터 리스트로 만듭니다.
            logits_np = attn_weights.detach().cpu().numpy()
            rows_list = [row for row in logits_np.reshape(-1, T)]

            try:
                # 2) 시리얼 통신으로 하드웨어 Softmax 수행
                # 타임아웃을 넉넉하게 줍니다 (행렬이 크면 시간이 걸릴 수 있음)
                probs_list = softmax_batch(
                    self.ser, rows_list, pad_value=-32.0, timeout_s=10.0
                )

                # 3) 결과를 다시 텐서 형태로 복원
                probs_flat = np.vstack(probs_list)
                attn_weights = torch.tensor(
                    probs_flat.reshape(B, H, T, T),
                    dtype=query.dtype,
                    device=query.device,
                )
            except Exception as e:
                print(f"[HW Error in Layer {self.layer_idx}] {e}")
                # 하드웨어 통신 에러 시, 소프트웨어 Softmax로 대체(Fallback)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 기본 소프트웨어 Softmax (Baseline)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # ============================================================

        # [시각화용] 계산된 Attention Weights 저장
        self.last_attn = attn_weights.detach().cpu()

        # Dropout 및 Head Mask 적용 (기존 로직)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


def replace_gpt2_attention(model, config_updates):
    """
    모델 내의 모든 GPT2Attention 모듈을 커스텀 모듈로 교체합니다.
    """
    for name, module in model.named_modules():
        if isinstance(module, GPT2Attention):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
                child_name = name.split(".")[-1]
            else:
                parent = model
                child_name = name

            config = copy.deepcopy(module.config)
            for k, v in config_updates.items():
                setattr(config, k, v)

            new_module = GPT2AttentionSoftmaxApprox(
                config=config,
                is_cross_attention=module.is_cross_attention,
                layer_idx=module.layer_idx,
            )

            new_module.load_state_dict(module.state_dict(), strict=False)
            new_module.to(next(module.parameters()).device)

            setattr(parent, child_name, new_module)

    print(f"Replaced Attention layers with HW-Approx version.")


def build_model_GPT2(ser=None, model_name="gpt2"):
    print(f"[Init] Loading {model_name} tokenizer & baseline model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    approx_model = copy.deepcopy(base_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Moving models to device: {device}")
    base_model.to(device).eval()
    approx_model.to(device).eval()

    config_updates = {"use_approx": True, "ser": ser}

    print("[Init] Replacing Attention modules...")
    replace_gpt2_attention(approx_model, config_updates)

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
