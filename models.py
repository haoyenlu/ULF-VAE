import torch
import torch.nn as nn
from torchsummary import summary

class VariationalAutoencoderConv(nn.Module):
    def __init__(self,seq_len,feat_dim,latent_dim,hidden_dim = 50):
        super(VariationalAutoencoderConv,self).__init__()

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

        self.dense_mean = nn.Linear(self.seq_len // 64 * self.hidden_dim,self.latent_dim)
        self.dense_var = nn.Linear(self.seq_len // 64 * self.hidden_dim,self.latent_dim)

        self.first_decoder_dense = nn.Sequential(
            nn.Linear(self.latent_dim,self.seq_len // 64 * self.hidden_dim),
            nn.ReLU()
        )

        self.last_decoder_dense = nn.Linear(self.seq_len * self.hidden_dim, self.seq_len * self.feat_dim)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def _make_encoder_block(self,in_channel,out_channel,kernel=3,downsample=False):
        block = []

        block.append(nn.Conv1d(in_channel,out_channel,kernel,padding="same"))
        block.append(nn.LeakyReLU(0.2))
        if downsample:
            block.append(nn.AvgPool1d(kernel_size=3,padding=1,stride=2))

        return nn.Sequential(*block)
    

    def _make_decoder_block(self,in_channel,out_channel,kernel=3,upsample=False):
        block = []
        if upsample:
            block.append(nn.Upsample(scale_factor=2))
        
        block.append(nn.Conv1d(in_channel,out_channel,kernel,padding="same"))
        block.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*block)

    def _get_encoder(self):

        model = nn.Sequential(
            self._make_encoder_block(self.feat_dim,self.hidden_dim,kernel=3,downsample=True),
            self._make_encoder_block(self.hidden_dim,self.hidden_dim,kernel=3,downsample=True),
            self._make_encoder_block(self.hidden_dim,self.hidden_dim,kernel=3,downsample=True),
            self._make_encoder_block(self.hidden_dim,self.hidden_dim,kernel=3,downsample=True),
            self._make_encoder_block(self.hidden_dim,self.hidden_dim,kernel=3,downsample=True),
            self._make_encoder_block(self.hidden_dim,self.hidden_dim,kernel=3,downsample=True),
            nn.Flatten()
        )

        return model
    
    def _get_decoder(self):
        model = nn.Sequential(
            self._make_decoder_block(self.hidden_dim,self.hidden_dim,kernel=3,upsample=True),
            self._make_decoder_block(self.hidden_dim,self.hidden_dim,kernel=3,upsample=True),
            self._make_decoder_block(self.hidden_dim,self.hidden_dim,kernel=3,upsample=True),
            self._make_decoder_block(self.hidden_dim,self.hidden_dim,kernel=3,upsample=True),
            self._make_decoder_block(self.hidden_dim,self.hidden_dim,kernel=3,upsample=True),
            self._make_decoder_block(self.hidden_dim,self.hidden_dim,kernel=3,upsample=True),
            nn.Flatten()
        )

        return model


    def sampling(self,z_mean,z_log_var):
        return torch.normal(mean=z_mean,std=torch.exp(0.5*z_log_var))
    
    def summary(self):
        summary(self.encoder,(self.feat_dim,self.seq_len))
        summary(self.decoder,(1,self.latent_dim))



    def forward(self,X): # shape: (B,feats,seq_len)
        z_mean, z_log_var = self.encoding(X)
        z = self.sampling(z_mean,z_log_var)
        _z = self.decoding(z)

        return _z, (z_mean,z_log_var)

    def encoding(self,X):
        _x = self.encoder(X)
        z_mean = self.dense_mean(_x)
        z_log_var = self.dense_var(_x)
        return z_mean, z_log_var

    def decoding(self,z):
        batch_size = z.shape[0]
        _z = self.first_decoder_dense(z)
        _z = torch.reshape(_z,(batch_size,self.hidden_dim,-1))
        _z = self.decoder(_z)
        _z = self.last_decoder_dense(_z)
        _z = torch.reshape(_z,(batch_size,self.feat_dim,self.seq_len))
        return _z

    
    def get_prior_samples(self,num_samples):
        Z = torch.randn(num_samples,self.latent_dim)
        _z = self.decoding(Z)
        return _z

    def get_prior_samples_given_Z(self,Z):
        return self.decoding(Z)

 
