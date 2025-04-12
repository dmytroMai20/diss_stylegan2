import torch
from generator import Generator

class EMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = self.clone_model(model)
        self.decay = decay

    def clone_model(self, model):
        ema_model = Generator(*model.args).to(next(model.parameters()).device)
        ema_model.load_state_dict(model.state_dict())
        for param in ema_model.parameters():
            param.requires_grad_(False)
        ema_model.eval()
        return ema_model

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.copy_(self.decay * ema_param + (1. - self.decay) * param)