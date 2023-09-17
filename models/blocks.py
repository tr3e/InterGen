from .layers import *


class TransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out
