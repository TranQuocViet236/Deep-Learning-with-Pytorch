from libs import *


class L2Norm(nn.Module):
    def __init__(self, input_channel=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channel))
        self.scale = scale
        self.reset_parameters()

        self.eps = 1e-0


    def reset_parameters(self):
        nn.init.constant_(self.weight, self.scale)


    def forward(self, x):
        # x.size() = (batch_size, channel, height, width)
        #l2 norm
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        #wright.size() = (512) -> (1, 512) ->
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        return weights*x


