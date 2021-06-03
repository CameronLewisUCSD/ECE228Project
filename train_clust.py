import torch
import numpy as np
from sklearn.mixture import GaussianMixture

def clustTrain(model,dataloaders=None,print_every=10, latentSpaceArray=None):
    #The sklearn GMM object can only use data in memory. 
    #We need the data to be processed to the latent space.
    #We can avoid loading everything into memory by using the warm_start parameter to continue training our object with multiple calls of fit.
    model.to(torch.device('cpu'))
    if latentSpaceArray is not None:
        print("start training GMM")
        model.gmm = GaussianMixture(
                  n_components=8,
                  max_iter=1000,
                  n_init=100)
        model.gmm.fit(latentSpaceArray)
        print("finished training GMM")
    else:
        model.gmm = GaussianMixture(
                  n_components=8,
                  max_iter=1000,
                  n_init=100, #this is ignored with warm_start=True
                  warm_start=True)
        for idx,(y_index,batch) in enumerate(dataloaders[0]):
            _, latent_batch = model(batch)
            model.gmm.fit(latent_batch.detach().cpu())
            if idx% print_every ==0:
                #possibly do some visualisations
                #this is mid training, it would be better to do this over multiple epochs really.
                print("idx:", idx)
    #we return the trained model.gmm on the dataset.
    return model

#use the results of this to get the model.gmm.predict(z_array) and tsne(z_array) for final visualizations
def getLatentFeatureSpaceDataset(model,dataloader,double=True):
    model.to(torch.device('cpu'))
    z_array = np.zeros((dataloader.len, 9), dtype=np.float64)
    #labels_prev = np.zeros((len(dataloaders[2].dataset), 9), dtype=np.float32)
    bsz = dataloader.batch_size
    for b, batch in enumerate(dataloader):
                _, batch = batch
                x = batch.to(torch.device('cpu'))
                _, z = model(x)
                z_array[b * bsz:(b*bsz) + x.size(0), :] = z.detach().cpu().numpy()
    return z_array