import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        # Linear transformations for query, key, value
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # Output projection
        self.out_proj = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        B, N, F = x.size()  # B: batch size, N: sequence length, F: feature dimension

        # Linear transformations
        queries = self.query(x)  # shape: (B, N, in_dim)
        keys = self.key(x)  # shape: (B, N, in_dim)
        values = self.value(x)  # shape: (B, N, in_dim)

        # Reshape queries, keys, values for multi-head attention
        queries = queries.view(B, N, self.num_heads, self.head_dim).transpose(1,
                                                                              2)  # shape: (B, num_heads, N, head_dim)
        keys = keys.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (B, num_heads, N, head_dim)
        values = values.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (B, num_heads, N, head_dim)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (
                self.head_dim ** 0.5)  # shape: (B, num_heads, N, N)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # shape: (B, num_heads, N, N)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        attention_output = torch.matmul(attention_weights, values)  # shape: (B, num_heads, N, head_dim)

        # Reshape attention output and concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, N,
                                                                              self.in_dim)  # shape: (B, N, in_dim)

        # Apply final linear transformation and combine with original input
        output = self.out_proj(attention_output)  # shape: (B, N, in_dim)

        # Residual connection
        output = output + x  # Combine with original input

        return output


# # Example usage:
# batch_size = 16
# num_matches = 1000
# feature_length = 72
#
# input_tensor = torch.randn(batch_size, num_matches, feature_length)  # Example input
#
# # Create self-attention module
# self_attention = SelfAttention(feature_length, num_heads=8)
#
# # Apply self-attention
# output = self_attention(input_tensor)

# print("Output shape:", output.shape)  # Output shape should be (16, 1000, 72)
