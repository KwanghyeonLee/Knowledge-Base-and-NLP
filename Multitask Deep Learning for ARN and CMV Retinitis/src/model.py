from turtle import forward
import torch
import torch.nn as nn
from torch.autograd import Function

class SingleTaskMLP(nn.Module):
    def __init__(self, hidden_unit = 128, class_type = 1):
        '''
        class_type : 1 or 2
            1 : ARN detection model
            2 : CMV retinitis detection model
        '''
        super().__init__()
        self.class_type = class_type
        
        self.in_dim = 15 # Feature dimension
        self.out_dim = 1
        
        self.fc1 = nn.Linear(self.in_dim, hidden_unit)
        self.fc2 = nn.Linear(hidden_unit, self.out_dim)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = x.reshape(-1, self.in_dim)
        a = self.tanh(self.fc1(x))
        out = self.sigmoid(self.fc2(a))
        return out


class FullySharedMTL(nn.Module):
    def __init__(self, hidden_unit = 128):
        super().__init__()
        
        self.in_dim = 15
        self.out_dim = 1

        self.shared_fc = nn.Linear(self.in_dim, hidden_unit)
        self.fc1 = nn.Linear(hidden_unit, self.out_dim)
        self.fc2 = nn.Linear(hidden_unit, self.out_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.reshape(-1, self.in_dim)
        a = self.tanh(self.shared_fc(x))
        out1 = self.sigmoid(self.fc1(a))
        out2 = self.sigmoid(self.fc2(a))
        return out1, out2

class SharedPrivateMTL(nn.Module):
    def __init__(self, hidden_unit = 128):
        super().__init__()
        
        self.in_dim = 15
        self.out_dim = 1

        self.shared_fc = nn.Linear(self.in_dim, hidden_unit)
        self.private_fc_1 = nn.Linear(self.in_dim, hidden_unit)
        self.private_fc_2 = nn.Linear(self.in_dim, hidden_unit)
        self.fc1 = nn.Linear(2 * hidden_unit, self.out_dim)
        self.fc2 = nn.Linear(2 * hidden_unit, self.out_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.reshape(-1, self.in_dim)
        a_shared = self.tanh(self.shared_fc(x))
        a_private_1 = self.tanh(self.private_fc_1(x))
        a_private_2 = self.tanh(self.private_fc_2(x))
        out1 = self.sigmoid(self.fc1(torch.cat((a_shared, a_private_1), dim=1)))
        out2 = self.sigmoid(self.fc2(torch.cat((a_shared, a_private_2), dim=1)))
        return out1, out2

class GradiantReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.lambd
        lambd = grad_output.new_tensor(lambd)
        dx = -lambd * grad_output
        return dx, None

class GradiantReverseLayer(nn.Module):
    def __init__(self, lambd=1):
        super().__init__()
        self.lambd = lambd
    
    def forward(self, x):
        return GradiantReverse.apply(x, self.lambd)


class AdversarialMTL(nn.Module):
    def __init__(self, hidden_unit = 128):
        super().__init__()
        
        self.in_dim = 15
        self.out_dim = 1

        self.shared_fc = nn.Linear(self.in_dim, hidden_unit)
        self.private_fc_1 = nn.Linear(self.in_dim, hidden_unit)
        self.private_fc_2 = nn.Linear(self.in_dim, hidden_unit)

        self.discriminator = nn.Sequential(
            GradiantReverseLayer(),
            nn.Linear(hidden_unit, 3) # 3 classes
        )

        self.fc1 = nn.Linear(2 * hidden_unit, self.out_dim)
        self.fc2 = nn.Linear(2 * hidden_unit, self.out_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(-1, self.in_dim)
        a_shared = self.tanh(self.shared_fc(x))
        a_private_1 = self.tanh(self.private_fc_1(x))
        a_private_2 = self.tanh(self.private_fc_2(x))

        adversarial_out = self.discriminator(a_shared)

        out1 = self.sigmoid(self.fc1(torch.cat((a_shared, a_private_1), dim=1)))
        out2 = self.sigmoid(self.fc2(torch.cat((a_shared, a_private_2), dim=1)))
        return out1, out2, adversarial_out, (a_shared, a_private_1, a_private_2)