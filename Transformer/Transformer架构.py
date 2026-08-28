import torch
import torch.nn as nn
import math
import numpy as np

"""
Transformer架构的核心就是：
QK 相乘：算“该关注谁”
softmax：把关注程度变成权重
权重乘 V：从被关注的词那里取信息
多头：从多个角度并行关注
W_o：把多个角度的信息整合成最终输出
先把语言转换成为向量，确定每个词之间的关系，然后transformer里面有两种形式，自注意力 Self-Attention：Q、K、V 都来自同一句话/同一序列。
多头注意力 Multi-Head Attention：把注意力拆成多个头，并行从多个角度看关系。
这个架构里还有一个很重要的东西叫mask，就是为了避免大模型提前知道答案，因为每次生成一个token之后，都要根据这个token
和之前句子的联系进行下一步补充，但大模型的输出有超前性，会一下子生成很多后文，然后这个mask的作用就是把后面词语的权重降到0，避免
大模型提前知道答案影响后面的生成。

/第一次修改之后的理解
语言先被切成 token，再转换成向量。Transformer 会在每一层里通过注意力机制动态计算 token 之间的关系。

自注意力是让同一句话里的 token 彼此建立联系；交叉注意力是让解码器当前生成内容去关注编码器的输入内容。多头注意力不是另一种和自注意力并列的机制，而是把注意力拆成多个头，从多个角度并行捕捉关系。

mask 的作用是限制模型能看到的信息，尤其是在 Decoder 里防止当前位置看到未来 token。训练时模型会并行处理完整序列，如果没有 mask，就可能在预测当前 token 时偷看后面的正确答案。mask 会让未来 token 的注意力权重接近 0。
"""

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """

    class PositionalEncoding(nn.Module):
        """
        为输入序列的词嵌入向量添加位置编码。
        """

        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            # 创建一个足够长的位置编码矩阵
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

            # pe (positional encoding) 的大小为 (max_len, d_model)
            pe = torch.zeros(max_len, d_model)

            # 偶数维度使用 sin, 奇数维度使用 cos
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            # 将 pe 注册为 buffer，这样它就不会被视为模型参数，但会随模型移动（例如 to(device)）
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x.size(1) 是当前输入的序列长度
            # 将位置编码加到输入向量上
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 和输出的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 1. 计算注意力得分 (QK^T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. 应用掩码 (如果提供)
        if mask is not None:
            # 将掩码中为 0 的位置设置为一个非常小的负数，这样 softmax 后会接近 0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 3. 计算注意力权重 (Softmax)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 4. 加权求和 (权重 * V)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # 将输入 x 的形状从 (batch_size, seq_length, d_model)
        # 变换为 (batch_size, num_heads, seq_length, d_k)
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # 将输入 x 的形状从 (batch_size, num_heads, seq_length, d_k)
        # 变回 (batch_size, seq_length, d_model)
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 1. 对 Q, K, V 进行线性变换
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 2. 计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合并多头输出并进行最终的线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络模块
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形状: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # 最终输出形状: (batch_size, seq_len, d_model)
        return x


# --- 编码器核心层 ---

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 残差连接与层归一化将在 3.1.2.4 节中详细解释
        # 1. 多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

# --- 解码器核心层 ---

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads) # 待实现
        self.cross_attn = MultiHeadAttention(d_model, num_heads) # 待实现
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout) # 待实现
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. 掩码多头自注意力 (对自己)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 交叉注意力 (对编码器输出)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

if __name__ == "__main__":
    torch.manual_seed(42)

    vocab_size = 10
    d_model = 8
    num_heads = 2
    d_ff = 32
    dropout = 0.1

    # 假设一句话有4个token，它们的编号分别是 2, 5, 3, 7
    token_ids = torch.tensor([[2, 5, 3, 7]])

    embedding = nn.Embedding(vocab_size, d_model)
    encoder = EncoderLayer(d_model, num_heads, d_ff, dropout)

    x = embedding(token_ids)

    mask = torch.ones(1, 1, 1, 4)

    encoder.eval()
    with torch.no_grad():
        output = encoder(x, mask)

    print("token ids:")
    print(token_ids)

    print("\nEmbedding 后的向量形状:")
    print(x.shape)

    print("\nEmbedding 后的向量:")
    print(x)

    print("\n经过 EncoderLayer 后的输出向量形状:")
    print(output.shape)

    print("\n经过 EncoderLayer 后的输出向量:")
    print(output)