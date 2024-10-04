import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class PandasDataSet_WithEncoder(TensorDataset):
    def __init__(self, X, label, s=None, encoder=None, batch_size=512, device='cpu'):
        X = self._df_to_tensor(X).to(device)
        label = self._df_to_tensor(label).to(device)
        s = self._df_to_tensor(s).to(device) if s is not None else s
        
        if encoder is not None:
            encoded_X = []
            for i in range(0, X.size(0), batch_size):
                batch_X = X[i:i+batch_size]
                with torch.no_grad():
                    encoded = encoder(batch_X)
                    # if it is VFAE, then use z
                    if isinstance(encoded, tuple):
                        encoded = encoded[0]
                    encoded_X.append(encoded.detach().to(device))
            X = torch.cat(encoded_X, dim=0)
        
        tensors = (X, label.squeeze().long(), s.squeeze().long())
        super(PandasDataSet_WithEncoder, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame("dummy")
        return torch.from_numpy(df.values).float()

class PandasDataSet_LRTest(TensorDataset):
    def __init__(self, X, label, encoder=None, batch_size=512, device='cpu'):
        X = self._df_to_tensor(X).to(device)
        label = self._df_to_tensor(label).to(device)
        
        if encoder is not None:
            encoded_X = []
            for i in range(0, X.size(0), batch_size):
                batch_X = X[i:i+batch_size]
                with torch.no_grad():
                    encoded_X.append(encoder(batch_X).to(device))
            X = torch.cat(encoded_X, dim=0)
        
        tensors = (X, label.squeeze().long())
        super(PandasDataSet_LRTest, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame("dummy")
        return torch.from_numpy(df.values).float()

class PandasDataSet(TensorDataset):
    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)
        

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame("dummy")
        return torch.from_numpy(df.values).float()

def InfiniteDataLoader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True):
    while True:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        for data in data_loader:
            yield data