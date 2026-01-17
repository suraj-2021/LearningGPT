class GELU(nn.Module):
    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate
    
    def forward(self, x):
        if self.approximate == "tanh":
            return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2/3.14159)) * (x + 0.044715 * torch.pow(x, 3))))
        return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
