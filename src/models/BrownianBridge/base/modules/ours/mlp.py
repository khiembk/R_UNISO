import torch
from torch import nn

class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x
    
class MLP(nn.Module):

    def __init__(
            self,
            input_dim=2,
            index_dim=1,
            hidden_dim=128,
            act=Swish(),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim
        self.act = act
        self.y_dim = 1
        self.main = nn.Sequential(
            nn.Linear(input_dim + index_dim + 2*self.y_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, input_dim),
        )
        """
        self.main = nn.Sequential(
            nn.Linear(input_dim + index_dim + self.y_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )
        """

    def forward(self, x_t, t, y_high, y_low):
        # init
        
        sz = x_t.size()
        x_t = x_t.view(-1, self.input_dim)
        t = t.view(-1, self.index_dim).float()
        y_high = y_high.view(-1, self.y_dim).float()
        y_low = y_low.view(-1, self.y_dim).float()
       
        # forward
        h = torch.cat([x_t, t, y_high, y_low], dim=1)  # concat
        output = self.main(h)  # forward
        return output.view(*sz)


class task_classifier_MLP(nn.Module):

    def __init__(
            self,
            input_dim=2,
            index_dim=1,
            hidden_dim=128,
            task_dim = 128,
            act=Swish(),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.act = act
        self.y_dim = 1
        self.main = nn.Sequential(
            nn.Linear(input_dim + index_dim + 2*self.y_dim + self.task_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, input_dim),
        )
        """
        self.main = nn.Sequential(
            nn.Linear(input_dim + index_dim + self.y_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )
        """

    def forward(self, x_t, t, y_high, y_low, task_info = None):
        # init
        
        sz = x_t.size()
        x_t = x_t.view(-1, self.input_dim)
        t = t.view(-1, self.index_dim).float()
        y_high = y_high.view(-1, self.y_dim).float()
        y_low = y_low.view(-1, self.y_dim).float()
        batch_size = y_low.shape[0]
        if task_info is not None:
           task_info = task_info.view(-1, self.task_dim).float()
        else: 
           task_info = torch.zeros((batch_size, self.task_dim)) 


        # forward
        h = torch.cat([x_t, t, y_high, y_low, task_info], dim=1)  # concat
        output = self.main(h)  # forward
        return output.view(*sz)