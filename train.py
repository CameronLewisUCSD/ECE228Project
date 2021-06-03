#In this file we will place the training functions of our network
#assume model and 
#epoch loop
import torch
import numpy as np


print_every=100
def pretrain(model,dataloaders,device,lr=0.001, epochs=10,absoluteLossThresh=None):
    loss_train_epoch=[]
    loss_validation_epoch=[]
    loss_testing_epoch=[]
    optim = torch.optim.Adam(model.parameters())
    loss = torch.nn.MSELoss()
    for e in range(epochs):
        #loop over train
        loss_train=[]
        loss_validation=[]
        loss_testing=[]
        model.train(True)
        for idx,batch in enumerate(dataloaders[0]):

            with torch.set_grad_enabled(True):
                _,batch= batch
                batch=batch.to(device)
                x_rec, _=model(batch)
                x=batch
                loss_mse = loss(x_rec, x)
                optim.zero_grad()
                loss_mse.backward()
                optim.step()
                loss_train.append(loss_mse.cpu().detach())
            
            #early stop loss
            if absoluteLossThresh is not None and loss_train[idx] < absoluteLossThresh:
                break


            if idx %print_every==0:
                #store output
                print("epoch:  ",e," train loss: ",loss_train[idx])
            pass

        #loop over valid

        model.train(False)
        loss = torch.nn.MSELoss()
        for idx,batch in enumerate(dataloaders[1]):
#             outValid = model(batch)
#             loss_valid = loss.backward()
            with torch.no_grad():
                _,batch= batch
                batch=batch.to(device)
                x_rec, _=model(batch)
                x=batch
                loss_mse = loss(x_rec, x)
                loss_validation.append(loss_mse.cpu().detach())
            if absoluteLossThresh is not None and loss_validation[idx] < absoluteLossThresh:
                break
            if idx %print_every==0:
                #store output
                print("epoch:  ",e," val loss: ",loss_validation[idx])
                
            pass
        pass
    
        #loop over train
        model.eval()                 
        for idx,batch in enumerate(dataloaders[2]): #dataloaders[2] = testing set
            with torch.no_grad():
                _,batch= batch
                batch=batch.to(device)
                x_rec, _=model(batch)
                x=batch
                loss_mse = loss(x_rec, x)
                loss_testing.append(loss_mse.cpu().detach())
            if absoluteLossThresh is not None and loss_testing[idx] < absoluteLossThresh:
                break
            if idx %print_every==0:
                #store output
                print("epoch:  ",e," val loss: ",loss_testing[idx])
            pass
    
 
        loss_train_epoch.append(sum(loss_train)/len(loss_train))
        loss_validation_epoch.append(sum(loss_validation)/len(loss_validation))
        loss_testing_epoch.append(sum(loss_testing)/len(loss_testing))
        print('\n========================================================')
        print('train loss', loss_train_epoch[e],'val_loss ', loss_validation_epoch[e])
        print('========================================================\n')
        
    return(loss_train_epoch,loss_validation_epoch,loss_testing_epoch)


    