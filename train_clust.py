import torch
import numpy as np
from sklearn.mixture import GaussianMixture

def clustTrain(model,dataloaders,print_every=10):
    #The sklearn GMM object can only use data in memory. 
    #We need the data to be processed to the latent space.
    #We can avoid loading everything into memory by using the warm_start parameter to continue training our object with multiple calls of fit.
    
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
def testPreditions(model,dataloaders):
    totalPreds = np.zeros(len(datloaders[2]))
    for idx,(y_index,batch) in enumerate(dataloaders[2]):
        _, latent_batch = model(batch)
        totalPred[y_index] = model.gmm.predict(batch)
        if idx% print_every ==0:
            #possibly do some visualisations
            #this is predictions, it would be better to do this over multiple epochs really.
            print("idx:", idx)
    
    return totalPreds
