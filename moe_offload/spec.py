"""Model shape and memory accounting for a routed-MoE decoder (DeepSeek/Solar-style: shared + routed SwiGLU experts)."""
import json, os
from dataclasses import dataclass

BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    experts: int
    topk: int
    expert_ffn: int
    shared_experts: int
    vocab: int
    dtype_bytes: int = 2

    @classmethod
    def from_hf_config(cls, path, name=None):
        c = json.load(open(os.path.join(path, "config.json") if os.path.isdir(path) else path))
        return cls(name=name or c.get("model_type", "moe"), hidden=c["hidden_size"], layers=c["num_hidden_layers"],
                   heads=c["num_attention_heads"], kv_heads=c["num_key_value_heads"],
                   head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
                   experts=c["n_routed_experts"], topk=c["num_experts_per_tok"], expert_ffn=c["moe_intermediate_size"],
                   shared_experts=c.get("n_shared_experts", 0), vocab=c["vocab_size"],
                   dtype_bytes=BYTES.get(c.get("torch_dtype", "bfloat16"), 2))

    # ---- bytes ----
    @property
    def expert_bytes(self):            # gate + up + down
        return 3 * self.hidden * self.expert_ffn * self.dtype_bytes

    @property
    def attn_bytes(self):              # q, k, v, o
        q = self.heads * self.head_dim; kv = self.kv_heads * self.head_dim
        return (self.hidden * q * 2 + self.hidden * kv * 2) * self.dtype_bytes

    @property
    def dense_layer_bytes(self):       # attention + shared experts + router + norms
        return (self.attn_bytes + self.shared_experts * self.expert_bytes
                + self.experts * self.hidden * self.dtype_bytes + 2 * self.hidden * self.dtype_bytes)

    @property
    def head_bytes(self):              # lm_head (embeddings stay on the host)
        return self.vocab * self.hidden * self.dtype_bytes

    def kv_bytes(self, ctx):
        return self.layers * 2 * self.kv_heads * self.head_dim * ctx * self.dtype_bytes

    @property
    def total_expert_bytes(self):
        return self.layers * self.experts * self.expert_bytes
