import torch
import torch.nn as nn

# 재현성을 위한 seed 고정
torch.manual_seed(123)

# 교재 예제 문장 기반 입력 임베딩
inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55], # step
])

tokens = ["Your", "journey", "starts", "with", "one", "step"]
print("inputs.shape:", inputs.shape)

def get_attention_scores(inputs, query_index):
    """
    inputs: [num_tokens, embedding_dim]
    query_index: 기준이 되는 토큰 위치
    return: attn_scores [num_tokens]
    """
    # TODO 1. query_index에 해당하는 query 벡터를 가져오세요.
    query = inputs[query_index]

    # TODO 2. attention score를 저장할 빈 텐서를 만드세요.
    attn_scores = torch.empty(inputs.shape[0])

    # TODO 3. 모든 입력 벡터와 query의 dot product를 계산하세요.
    for i, x_i in enumerate(inputs):
        attn_scores[i] = torch.dot(x_i, query)

    return attn_scores

scores = get_attention_scores(inputs, query_index=1)
print(scores)


expected_scores = torch.tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
assert scores.shape == torch.Size([6])
assert torch.allclose(scores, expected_scores, atol=1e-4)
print("Part 1 통과")


def get_context_vector(inputs, query_index):
    # TODO 1. attention score 계산
    attn_scores = get_attention_scores(inputs, query_index)

    # TODO 2. softmax로 attention weight 계산
    attn_weights = torch.softmax(attn_scores, dim=0)

    # TODO 3. weighted sum으로 context vector 계산
    context_vec = attn_weights @ inputs

    return context_vec, attn_weights

context_vec, attn_weights = get_context_vector(inputs, query_index=1)
print("attention weights:", attn_weights)
print("sum:", attn_weights.sum())
print("context vector:", context_vec)


assert attn_weights.shape == torch.Size([6])
assert torch.allclose(attn_weights.sum(), torch.tensor(1.0), atol=1e-6)
assert context_vec.shape == torch.Size([3])
expected_context = torch.tensor([0.4419, 0.6515, 0.5683])
assert torch.allclose(context_vec, expected_context, atol=1e-4)
print("Part 2 통과")


def self_attention_basic(inputs):
    """
    trainable weight가 없는 기본 self-attention 구현
    return: attn_weights [num_tokens, num_tokens], context_vecs [num_tokens, embedding_dim]
    """
    # TODO 1. attention score matrix 계산
    attn_scores = inputs @ inputs.T

    # TODO 2. 각 row마다 softmax 적용
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # TODO 3. context vector matrix 계산
    context_vecs = attn_weights @ inputs

    return attn_weights, context_vecs

attn_weights_basic, context_vecs_basic = self_attention_basic(inputs)
print("attn_weights_basic.shape:", attn_weights_basic.shape)
print("context_vecs_basic.shape:", context_vecs_basic.shape)
print("row sums:", attn_weights_basic.sum(dim=-1))


assert attn_weights_basic.shape == torch.Size([6, 6])
assert context_vecs_basic.shape == torch.Size([6, 3])
assert torch.allclose(attn_weights_basic.sum(dim=-1), torch.ones(6), atol=1e-6)
assert torch.allclose(context_vecs_basic[1], torch.tensor([0.4419, 0.6515, 0.5683]), atol=1e-4)
print("Part 3 통과")

def self_attention_qkv_manual(inputs, d_out=2):
    torch.manual_seed(123)
    d_in = inputs.shape[1]

    W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    # TODO 1. queries, keys, values 계산
    queries = inputs @ W_query
    keys = inputs @ W_key
    values = inputs @ W_value

    # TODO 2. attention scores 계산
    attn_scores = queries @ keys.T

    # TODO 3. scaled dot-product attention 적용
    d_k = keys.shape[-1]
    attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)

    # TODO 4. context vectors 계산
    context_vecs = attn_weights @ values

    return context_vecs, attn_weights, queries, keys, values

context_vecs_qkv, attn_weights_qkv, queries, keys, values = self_attention_qkv_manual(inputs)
print("queries.shape:", queries.shape)
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
print("attn_weights_qkv.shape:", attn_weights_qkv.shape)
print("context_vecs_qkv.shape:", context_vecs_qkv.shape)

assert queries.shape == torch.Size([6, 2])
assert keys.shape == torch.Size([6, 2])
assert values.shape == torch.Size([6, 2])
assert attn_weights_qkv.shape == torch.Size([6, 6])
assert context_vecs_qkv.shape == torch.Size([6, 2])
assert torch.allclose(attn_weights_qkv.sum(dim=-1), torch.ones(6), atol=1e-6)
print("Part 4 통과")


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # TODO 1. keys, queries, values 계산
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # TODO 2. attention scores 계산
        attn_scores = queries @ keys.T

        # TODO 3. attention weights 계산
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # TODO 4. context vectors 계산
        context_vecs = attn_weights @ values


        return context_vecs

torch.manual_seed(789)
sa = SelfAttention(d_in=3, d_out=2)
out_sa = sa(inputs)
print(out_sa)
print(out_sa.shape)

assert out_sa.shape == torch.Size([6, 2])
print("Part 5 통과")


def apply_causal_mask(attn_scores):
    context_length = attn_scores.shape[0]

    # TODO 1. upper triangular mask 생성
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()

    # TODO 2. mask 위치를 -inf로 채우기
    masked_scores = attn_scores.masked_fill(mask, -torch.inf)

    return masked_scores

attn_scores_qkv = queries @ keys.T
masked_scores = apply_causal_mask(attn_scores_qkv)
masked_weights = torch.softmax(masked_scores / keys.shape[-1]**0.5, dim=-1)
print(masked_weights)
print("row sums:", masked_weights.sum(dim=-1))

assert masked_weights.shape == torch.Size([6, 6])
assert torch.allclose(masked_weights.sum(dim=-1), torch.ones(6), atol=1e-6)
assert torch.all(masked_weights[0, 1:] == 0)
assert torch.all(masked_weights[1, 2:] == 0)
assert torch.all(masked_weights[2, 3:] == 0)
print("Part 6 통과")
def apply_attention_dropout(attn_weights, dropout_rate=0.5):
    torch.manual_seed(123)

    # TODO 1. Dropout 객체 생성
    dropout = nn.Dropout(dropout_rate)

    # TODO 2. attention weight에 dropout 적용
    dropped_weights = dropout(attn_weights)

    return dropped_weights

for p in [0.0, 0.1, 0.5]:
    dropped = apply_attention_dropout(masked_weights, dropout_rate=p)
    print(f"dropout={p}")
    print(dropped)

assert apply_attention_dropout(masked_weights, 0.0).shape == masked_weights.shape
print("Part 7 통과")

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # TODO 1. keys, queries, values 계산
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # TODO 2. attention scores 계산. 힌트: keys.transpose(1, 2)
        attn_scores = queries @ keys.transpose(1, 2)

        # TODO 3. causal mask 적용
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        # TODO 4. softmax 적용
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # TODO 5. dropout 적용
        attn_weights = self.dropout(attn_weights)

        # TODO 6. context vectors 계산
        context_vecs = attn_weights @ values
        return context_vecs

batch = torch.stack((inputs, inputs), dim=0)
torch.manual_seed(123)
ca = CausalAttention(d_in=3, d_out=2, context_length=batch.shape[1], dropout=0.0)
context_vecs_ca = ca(batch)
print(context_vecs_ca)
print(context_vecs_ca.shape)

assert context_vecs_ca.shape == torch.Size([2, 6, 2])
print("Part 8 통과")

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        # TODO. 각 head의 출력을 마지막 차원 기준으로 concat
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
mha_wrapper = MultiHeadAttentionWrapper(d_in=3, d_out=2, context_length=6, dropout=0.0, num_heads=2)
context_vecs_mha_wrapper = mha_wrapper(batch)
print(context_vecs_mha_wrapper)
print(context_vecs_mha_wrapper.shape)

assert context_vecs_mha_wrapper.shape == torch.Size([2, 6, 4])
print("Part 9 통과")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out은 num_heads로 나누어 떨어져야 합니다."
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # TODO 1. Q/K/V 계산: [b, num_tokens, d_out]
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # TODO 2. head 분리: [b, num_tokens, num_heads, head_dim]
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # TODO 3. transpose: [b, num_heads, num_tokens, head_dim]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # TODO 4. attention scores 계산: [b, num_heads, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(2, 3)

        # TODO 5. causal mask 적용
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        # TODO 6. softmax + dropout
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # TODO 7. context 계산 후 shape 복원
        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)

        # TODO 8. output projection
        context_vec = self.out_proj(context_vec)
        return context_vec

torch.manual_seed(123)
mha = MultiHeadAttention(d_in=3, d_out=4, context_length=6, dropout=0.0, num_heads=2)
context_vecs_mha = mha(batch)
print(context_vecs_mha)
print(context_vecs_mha.shape)

assert context_vecs_mha.shape == torch.Size([2, 6, 4])
print("Part 10 통과")

x = torch.randn(4, 8, 16)

model = MultiHeadAttention(
    d_in=16,
    d_out=16,
    context_length=8,
    dropout=0.1,
    num_heads=4
)

out = model(x)
print(out.shape)
assert out.shape == torch.Size([4, 8, 16])
print("최종 과제 기본 실행 통과")

x = torch.randn(4, 8, 16)

for h in [1, 2, 4]:
    model = MultiHeadAttention(
        d_in=16,
        d_out=16,
        context_length=8,
        dropout=0.1,
        num_heads=h
    )

    out = model(x)
    print(f"num_heads={h}, output shape={out.shape}")
    assert out.shape == torch.Size([4, 8, 16])

x = torch.randn(4, 8, 16)

torch.manual_seed(123)
model_drop0 = MultiHeadAttention(
    d_in=16,
    d_out=16,
    context_length=8,
    dropout=0.0,
    num_heads=4
)

torch.manual_seed(123)
model_drop1 = MultiHeadAttention(
    d_in=16,
    d_out=16,
    context_length=8,
    dropout=0.1,
    num_heads=4
)

out_drop0 = model_drop0(x)
out_drop1 = model_drop1(x)

print("dropout=0.0:", out_drop0.shape)
print("dropout=0.1:", out_drop1.shape)
print("두 결과가 같은가?:", torch.allclose(out_drop0, out_drop1))
