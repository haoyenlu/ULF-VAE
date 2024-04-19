import torch
import torch.nn as nn


from models import VariationalAutoencoderConv

class TimeVAE:
    def __init__(self,seq_len,feat_dim,latent_dim,hidden_dim,
                 lr = 1e-4, reconstruction_wt = 3.0, kl_wt=0.5,
                 max_iters=1000,
                 saveDir=None,ckptPath=None,prefix="T01"):

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.reconstruction_wt = reconstruction_wt
        self.lr = lr
        self.max_iters = max_iters
        self.kl_wt = kl_wt

        self.model = VariationalAutoencoderConv(
            seq_len = seq_len,
            feat_dim = feat_dim,
            latent_dim = latent_dim,
            hidden_dim=hidden_dim
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)

        self.saveDir = saveDir
        self.ckptPath = ckptPath
        self.prefix = prefix

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Train on {}".format(self.device))

        
        self.load_ckpt()
    
    def _get_reconstruction_loss(self,X,X_hat):
        err_all = torch.nn.functional.mse_loss(X_hat,X)
        return err_all

    def train(self,dataloader):
        data = self.get_infinite_batch(dataloader)

        # self.model.summary()
        self.model.to(self.device)
        self.model.train()

        for iter in range(self.max_iters):
            self.model.zero_grad()

            X = torch.autograd.Variable(data.__next__()).float().to(self.device)
            X_hat , (z_mean,z_log_var) = self.model(X)

            reconstruction_loss = self._get_reconstruction_loss(X,X_hat)

            kl_loss = (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
            kl_loss = torch.mean(kl_loss)

            total_loss = self.kl_wt * kl_loss + self.reconstruction_wt * reconstruction_loss
            total_loss.backward()
            self.optimizer.step()

            print(f"Iteration:{iter+1}/{self.max_iters},KL Loss:{kl_loss},Reconstruction Loss:{reconstruction_loss}")

            if (iter+1) % 20 == 0:
                self.save_model()
            
            torch.cuda.empty_cache()
        
        self.save_model()
        print("Finished Training")


    def save_model(self):
        torch.save(self.model.state_dict(),
                   f"{self.saveDir}/TimeVAE_{self.prefix}_{self.latent_dim}_ckpt.pth")
        
    
    def load_ckpt(self):
        if self.ckptPath:
            print("Loading Checkpoint...")
            ckpt = torch.load(self.ckptPath,map_location=self.device)
            self.model.load_state_dict(ckpt)

    def generate_samples(self,sample_sizes):
        samples = self.model.get_prior_samples(sample_sizes)
        return samples.detach().cpu().numpy()
    

    def get_infinite_batch(self,dataloader):
        while True:
            for data in dataloader:
                yield data



