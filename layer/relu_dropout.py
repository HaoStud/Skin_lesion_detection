import torch

# https://gist.github.com/vadimkantorov/360ece06de4fd2641fa9ed1085f76d48
class ReLUDropout(torch.nn.Dropout):
    def forward(self, input):
        return relu_dropout(input, p = self.p, training = self.training, inplace = self.inplace)

def relu_dropout(x, p = 0, inplace = False, training = False):
    if not training or p == 0:
        return x.clamp_(min = 0) if inplace else x.clamp(min = 0)

    p1m = 1 - p
    mask = torch.rand_like(x) > p1m
    # mask = torch.bernoulli(torch.tensor(p, device = x.device, dtype = torch.float32).expand_as(x))
    mask |= (x < 0)
    return x.masked_fill_(mask, 0).div_(p1m) if inplace else x.masked_fill(mask, 0).div(p1m)