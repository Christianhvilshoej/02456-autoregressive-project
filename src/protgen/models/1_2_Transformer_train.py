import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent/'data'))

from make_data import *
import torch #For my text editor
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda')
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(1)




class decoder_only_layer(nn.TransformerEncoderLayer):
    ## We will reuse code from nn.TransformerEncoderLayer
    def __init__(self, d_model, nhead, seq_len = 128, device=None):
        super().__init__(d_model=d_model, nhead=nhead, batch_first=True,norm_first = True)
        # We will hard code it to be causal, and assume all data is of shape d_model X seq_len. Since shorter proteins are padded with 0s

        self.register_buffer("mask",
            torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool),
                diagonal=1))
        
            
    def forward(self, protein, padding_mask):
        return super().forward(
            protein,
            src_mask = self.mask,
            src_key_padding_mask=padding_mask,
            is_causal = True
        )      

class Transformer_model(nn.Module):
    def __init__(self,d_model,nhead,d_ff,vocab_size=23,seq_len = 128, device=None):
        super().__init__()
        self.device=device
        self.seq_embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.pos_embedding = nn.Embedding(seq_len, d_model,device=device)
        self.dol1=decoder_only_layer(d_model=d_model, nhead=nhead, seq_len=seq_len, device = device)
        self.dol2=decoder_only_layer(d_model=d_model, nhead=nhead, seq_len=seq_len, device = device)
        self.ff1 =nn.Linear(d_model,d_ff)
        self.relu=nn.ReLU()
        self.ff2 =nn.Linear(d_ff,23) # 20 amino acids + 1 EOS + BOS + PAD
        
    def forward(self, x):
        padding_mask = (x==0)
        positions = torch.arange(x.size(1), device=x.device)
        x=self.seq_embedding(x) + self.pos_embedding(positions)
        x=self.dol1(x, padding_mask)
        x=self.dol2(x, padding_mask)
        x=self.ff1(x)
        x=self.relu(x)
        x=self.ff2(x)
        
        return(x)

def main():
    
    train = ProteinDataset('../../../data/train.csv')
    val = ProteinDataset('../../../data/val.csv')
    X_train = train.tokens.float()
    X_val = val.tokens.float()
    TM = Transformer_model(d_model=128, nhead=4, d_ff=200,device=device).to(device)

    ## Setup for Train loop
    X_train = X_train.long()
    X_val = X_val.long()
    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32
    )

    test_dataset = TensorDataset(X_val)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32
    )

    ##Train loop

    optim = torch.optim.Adam(TM.parameters(), lr=0.0001)
    loss = nn.CrossEntropyLoss(ignore_index=0)

    epochs = 200
    best_val = 10000
    best_weights = None

    for epoch in range(epochs):
        # Training
        TM.train()
        total_loss = 0

        for (batch,) in train_loader:
            optim.zero_grad()

            batch = batch.to(device)
            X = batch[:, :-1]
            Y = batch[:, 1:]

            output_train = TM(X)
            Loss = loss(output_train.reshape(-1, 23), Y.reshape(-1))

            total_loss += Loss.item()

            Loss.backward()
            optim.step()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        TM.eval()
        total_val_loss = 0
        id_true = 0 #for checking %correct aa assignments
        id_total = 0 

        with torch.no_grad():
            for (batch,) in test_loader:
                batch = batch.to(device)

                X = batch[:, :-1]
                Y = batch[:, 1:]

                output_test = TM(X)
                predictions = output_test.argmax(dim=-1)
                id_mask = Y!=0
                id_true += ((predictions == Y)*id_mask).sum()
                id_total += id_mask.sum() 
                
                val_loss = loss(output_test.reshape(-1, 23), Y.reshape(-1))

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_weights = TM.state_dict()
            print(f'New best weights found: Epoch {epoch} validation error: {best_val}')
            

        if (epoch%10==0) or (epoch == epochs-1):
            print(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Validation Loss: {avg_val_loss:.4f} |"
                f"Validation %id: {id_true/id_total}")
            

    torch.save(best_weights,'1_2_Transformer_model_trained.pth')
    
if __name__=="__main__":
    main()