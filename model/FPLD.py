from .modules import *

class FLDNet(nn.Module):
    def __init__(self, input_size = 224, patch_size = 16, embed_dim = 32, num_heads = 8):
        super(FLDNet, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.multi_head_attention = MultiHeadAttnBlock(input_size, patch_size, embed_dim, num_heads)

        self.main_classifier = MainClassifier(input_size, 2, 32, 2)
        self.auxiliary_classifier = AuxiliaryClassifier(input_size, patch_size, 2, embed_dim)

    def forward(self, x):
        attn_map, attn_output = self.multi_head_attention(x)

        x = torch.cat((x, attn_map), dim=1)
        logits = self.main_classifier(x)
        auxiliary_logits = self.auxiliary_classifier(attn_output)

        return logits, auxiliary_logits
