import argparse
import numpy as np
import torch


from networks import TimeVAE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',help="Patient Dataset path",default=None)
    parser.add_argument('--ckptDir',help="Save Checkpoint Directory",default=None)
    parser.add_argument('--sampleDir',help="Save Sample Directory",default=None)
    parser.add_argument('--ckpt',help="Checkpoint path",default=None)
    parser.add_argument('--max_iter',type=int,default=1000)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--task',help="Task to train on",default='T01')
    parser.add_argument('--sample_size',type=int,help="Sample Size",default=200)


    ''' VAE model '''
    parser.add_argument('--latent_dim',type=int,help="Latent Dimension Size",default=10)
    parser.add_argument('--hidden_dim',type=int,default=50)
    parser.add_argument('--recon_wt',type=float,help="Reconstruction Weight",default=3)
    parser.add_argument('--kl_wt',type=float,help="KL divergence Loss",default=0.5)



    args = parser.parse_args()

    # dataset = PatientDataset(root = args.dir,csv_file = "ParticipantCharacteristics.xlsx")
    # train_data,seq_len = get_task_data(dataset,args.task)

    train_data = np.load(args.dir)
    feat,seq_len = train_data[0].shape
    dataloader = torch.utils.data.DataLoader(train_data,args.batch_size,shuffle=True)

    print(f"Train with {args.max_iter} iterations")
    print(f"Features:{feat},Sequence Length:{seq_len}")


    model = TimeVAE(seq_len = seq_len, feat_dim = feat, latent_dim= args.latent_dim,
                        hidden_dim = args.hidden_dim,max_iters= args.max_iter,
                        reconstruction_wt=args.recon_wt,kl_wt=args.kl_wt,
                        saveDir= args.ckptDir, ckptPath= args.ckpt, prefix=args.task )

    model.train(dataloader)
    sample = model.generate_samples(args.sample_size)
    np.save(f"{args.sampleDir}/VAE{args.latent_dim}_{args.task}_samples.npy",sample)
    

