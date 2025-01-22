import torch
import torch.utils.data
import torch.optim

from torchvision import datasets, transforms
from torchvision.models.efficientnet import efficientnet_v2_s
from torch.optim import Adam  # type: ignore

from tqdm import tqdm


device = torch.device('cuda:1')


def custom_efficientnet():
    model = efficientnet_v2_s(pretrained=True).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features, out_features=1, bias=True).to(device)

    return model


def train_step(model, loss_fn, optimizer, inputs, labels):
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = loss_fn(outputs, labels)

    loss.backward()
    optimizer.step()

    return loss.item()


def train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs):
    losses = []
    val_losses = []

    epoch_train_losses = []
    epoch_test_losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        model.train()
        for x_batch, y_batch in tqdm(train_loader, total=len(train_loader)): #iterate ove batches
            x_batch = x_batch.to(device) #move to gpu
            y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
            y_batch = y_batch.to(device) #move to gpu

            loss = train_step(model, loss_fn, optimizer, x_batch, y_batch)

            epoch_loss += loss / len(train_loader)
            losses.append(loss)
        
        epoch_train_losses.append(epoch_loss)
        print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

        #validation doesnt requires gradient
        with torch.no_grad():
            model.eval()
            cumulative_loss = 0
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
                y_batch = y_batch.to(device)

                #model to eval mode
                model.eval()

                yhat = model(x_batch)
                val_loss = loss_fn(yhat,y_batch)
                cumulative_loss += loss / len(test_loader)
                val_losses.append(val_loss.item())


            epoch_test_losses.append(cumulative_loss)
            print('Epoch : {}, val loss : {}'.format(epoch+1,cumulative_loss))  
            
            best_loss = min(epoch_test_losses)
            
            #save best model
            if cumulative_loss <= best_loss:
                best_model_wts = model.state_dict()
            
            # #early stopping
            # early_stopping_counter = 0
            # if cum_loss > best_loss:
            #   early_stopping_counter +=1

            # if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
            #   print("/nTerminating: early stopping")
            #   break #terminate training
    
    return best_model_wts, epoch_test_losses, epoch_train_losses, losses, val_losses


def main():
    model = custom_efficientnet()
    loss_fn = torch.nn.BCEWithLogitsLoss()    
    # optimizer = Adam(model.parameters(), lr=0.001)
    running_loss = 0.0


    train_data = datasets.ImageFolder('./small-frame-chunks', transform=transforms.ToTensor())

    generator = torch.Generator().manual_seed(42)
    split = int(0.8 * len(train_data))
    train_data, test_data = torch.utils.data.random_split(
        dataset=train_data,
        lengths=[split, len(train_data) - split],
        generator=generator
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=True)

    losses = []
    val_losses = []

    epoch_train_losses = []
    epoch_test_losses = []

    n_epochs = 10
    early_stopping_tolerance = 3
    early_stopping_threshold = 0.03

    print("Training FC")
    optimizer = Adam(model.classifier[-1].parameters(), lr=0.001)
    best_model_wts, epoch_test_losses, epoch_train_losses, losses, val_losses = train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs=5)
    model.load_state_dict(best_model_wts)

    print("Tuning all layers")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = Adam(model.parameters(), lr=0.001)
    best_model_wts, epoch_test_losses, epoch_train_losses, losses, val_losses = train(model, loss_fn, optimizer, train_loader, test_loader, n_epochs=20)
    model.load_state_dict(best_model_wts)

    #load best model
    model.load_state_dict(best_model_wts)

    print(epoch_test_losses)
    print(epoch_train_losses)

    import json

    with open('model.pth', 'wb') as f:
        torch.save(model, f)

    with open('epoch_test_losses.json', 'w') as f:
        f.write(json.dumps(epoch_test_losses))

    with open('epoch_train_losses.json', 'w') as f:
        f.write(json.dumps(epoch_train_losses))


if __name__ == '__main__':
    main()