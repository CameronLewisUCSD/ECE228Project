#In this file we will place the training functions of our network
#assume model and 
#epoch loop
import torch
def pretrain(model,datalaoders,epochs=50,lr=0.01):
    optim = torch.optim.Adam()
    loss = torch.losses.MSE()
    for e in range(epochs):
        #loop over train
        model.train(True)

        for idx,batch in enumerate(dataloaders[0])
            model(batch)
            loss.backward()
            optim.step()

            if idx %print_every==0:
                #store output


                loss_train.append(loss_valid.cpu().detach())

            pass

        #loop over valid

        model.train(False)
        for idx,batch in enumerate(dataloaders[1])
            outValid = model(batch)
            loss_valid = loss.backward()

            if idx %print_every==0:
                #store output
                loss_validation.append(loss_valid.cpu().detach())
            pass
        pass
    model.eval(True)
    for idx,batch in enumerate(dataloaders[2]) #dataloaders[2] = testing set
        model(batch)
        if idx %print_every==0:
                #store output
                loss_testing.append(loss_valid.cpu().detach())
        pass

    