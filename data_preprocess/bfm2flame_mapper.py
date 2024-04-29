import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class Mapper(nn.Module):
    def __init__(self, in_dim=144, hidden_dim=256, out_dim=403, layers=1):
        super(Mapper, self).__init__()
        self.maps = nn.ModuleList([])
        for i in range(layers):
            in_dim_ = in_dim if i == 0 else hidden_dim
            out_dim_ = out_dim if i == layers-1 else hidden_dim
            self.maps.append(nn.Linear(in_dim_, out_dim_, bias=True))
            if not i == layers-1:
                self.maps.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))    
        
        self.maps[-1].apply(self._zero_weights)

    def _zero_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight,0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        for layer in self.maps:
            x = layer(x)    
        return x

def train():
    torch.manual_seed(0)
    device = torch.device('cuda')
    save_path = os.path.join('assets','bfm2flame_mapper')
    os.makedirs(save_path,exist_ok=True)
    
    bfm_params = np.load('../portrait4d/data/ffhq_all_shape_n_motion_bfm_params.npy')
    flame_params = np.load('../portrait4d/data/ffhq_all_shape_n_motion_params.npy')
    bfm_params = torch.from_numpy(bfm_params).reshape(-1,bfm_params.shape[-1]).to(device)
    flame_params = torch.from_numpy(flame_params).reshape(-1,flame_params.shape[-1]).to(device)
    
    bfm_params_train = bfm_params[:130000]
    bfm_params_val = bfm_params[130000:]

    flame_params_train = flame_params[:130000]
    flame_params_val = flame_params[130000:]
    
    in_dim = bfm_params.shape[-1]
    out_dim = flame_params.shape[-1]
    
    print('Define models')
    model = Mapper(in_dim=in_dim, hidden_dim=512, out_dim=out_dim, layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()
    
    batchsize = 512
    iterations = 200001
    
    print('Start training')
    for iter in range(iterations):
        cur_idx = torch.randperm(len(bfm_params_train))[:batchsize]
        x = bfm_params_train[cur_idx]
        y = flame_params_train[cur_idx]
        
        model.train()
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if iter % 200 == 0:
            print(f'Train: Iter [{iter+1}/{iterations}], Loss: {loss.item():.4f}')
        
        if iter % 200 == 0:
            with torch.no_grad():
                x_val = bfm_params_val
                y_val = flame_params_val
                model.eval()
                outputs_val = model(x_val)
                loss = criterion(outputs_val, y_val)
                
                print(f'Validation: Iter [{iter+1}/{iterations}], Loss: {loss.item():.4f}')
        
        if iter % 50000 == 0:
            torch.save(model.state_dict(), os.path.join(save_path,f"model-iter{iter:06d}.pth"))

if __name__ == "__main__":
    train()
