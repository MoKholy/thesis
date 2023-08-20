from scripts.model import DeepLDA
from scripts.ldaloss import LinearDiscriminativeLoss
from scripts.dataset import load_dataset
import torch
import torch.optim as optim
import argparse 
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm 
import random
import os
import json
from tensorboardX import SummaryWriter

# set seed for reproducibility
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, train_loader, val_loader):

    # create summary writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # create loss function
    criterion = LinearDiscriminativeLoss(lambda_val=args.loss_lamda_val, eps=args.eps)
    # create optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    
    # create scheduler
    if args.scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.schedular_gamma)

    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.schedular_gamma, patience=10, verbose=True)
    else:
        raise NotImplementedError
    
    # save training args
    with open(os.path.join(args.model_save_loc, f"{args.model_name}_args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)


    # train model
    # set seed
    set_seed(args)
    # set device
    device = torch.device("cuda"+args.gpu_id if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_lda_loss = 0.0
    val_lda_loss = 0.0
    # train loop
    for epoch in range(args.epochs):
        
        # set model to train mode
        model.train()

        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for batch_index, (data, target) in enumerate(loop):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
            loop.set_description(f"Epoch [{epoch}/{args.epochs}]")
            loop.set_postfix(train_lda_loss=loss.item())
            train_lda_loss += loss.item()
        # print average loss for epoch
        print(f"Epoch [{epoch}/{args.epochs}] Average LDA Loss: {train_lda_loss/len(train_loader)}")
        # reset lda_loss
        train_lda_loss = 0.0
        # evaluate model

        # set model to eval mode
        model.eval()

        for batch_index, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loop.set_description(f"Epoch [{epoch}/{args.epochs}]")
            loop.set_postfix(val_lda_loss=loss.item())
            val_lda_loss += loss.item()
        # print average val lda loss for epoch
        print(f"Epoch [{epoch}/{args.epochs}] Average Val LDA Loss: {val_lda_loss/len(val_loader)}")
        # reset lda_loss
        val_lda_loss = 0.0

        # save model every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_save_loc, f"{args.model_name}_{epoch}.pth"))

            # save optimizer state
            torch.save(optimizer.state_dict(), os.path.join(args.model_save_loc, f"{args.model_name}_optimizer_{epoch}.pth"))

            # save scheduler state
            torch.save(scheduler.state_dict(), os.path.join(args.model_save_loc, f"{args.model_name}_scheduler_{epoch}.pth"))

            

        # save model at last epoch
        if epoch == args.epochs-1:
            torch.save(model.state_dict(), os.path.join(args.model_save_loc, f"{args.model_name}_last.pth"))

            # save optimizer state
            torch.save(optimizer.state_dict(), os.path.join(args.model_save_loc, f"{args.model_name}_optimizer_last.pth"))

            # save scheduler state
            torch.save(scheduler.state_dict(), os.path.join(args.model_save_loc, f"{args.model_name}_scheduler_last.pth"))


    # close writer
    writer.close()
    return "Training complete"
            

def test_dataloaders(train_loader, val_loader, test_loader):

    # not implemented yet
    pass
    # # check if last batch doesnt have equal number of elements as batch_size,
    # # if true -> set drop_last=True
    # prev_batch_size = 0
    # for batch_index, (data, target) in enumerate(train_loader):
    #     # brb



def test_parsing(args):
    # print all args
    for arg in vars(args):
        print(arg, getattr(args, arg))

def test_train(args):
    # test train function
    # load dataset
    train, val, test = load_dataset(name=args.dataset, stratify=args.stratify, \
                                    test_ratio= args.test_size, val_ratio=args.val_size, \
                                          random_state = args.seed)
    # create dataloaders 
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)

    # create model    
    model = DeepLDA(n_hidden=args.n_hidden, dropout=args.dropout, out_size=args.output_size, input_size=train[0][0].shape[0])


    # get optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    
    # get scheduler
    if args.scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.scheduler_gamma)
    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.scheduler_gamma, patience=10, verbose=True)
    else:
        raise NotImplementedError
    
    criterion = LinearDiscriminativeLoss(lambda_val=args.loss_lamda_val, eps=args.eps)

    # get just one batch
    overfit_data, overfit_target = next(iter(train_loader))

    model.train()
    for epoch in range(args.epochs):
        print(f"Epoch [{epoch}/{args.epochs}]")
        optimizer.zero_grad()
        output = model(overfit_data)
        loss = criterion(output, overfit_target)
        # print loss
        print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

    print("Training complete")

    


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()

    # arguments for seed 
    parser.add_argument("--seed", type=int, default=42)
    # arguments for loading dataset
    parser.add_argument("--dataset", type=str, default="correctly_labeled")
    parser.add_argument("--stratify", type=bool, default=True)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)

    # arguments for dataloader
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--drop_last", type=bool, default=False)


    # arguments for model instatioation
    parser.add_argument("--n_hidden", type=int, nargs="+", default=[64, 128, 256, 128, 64], help="Number of neurons per hidden layer")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--output_size", type=int, default=2)

    # arguments for training
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--log_dir", type=str, default="logs/")

    #LDA LOSS PARAMS
    parser.add_argument("--loss_lamda_val", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1.0)

        # optimizer
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
        # only if SGD 
    parser.add_argument("--nesterov", type=bool, default=True)
        # scheduler
    parser.add_argument("--scheduler", type=str, default="StepLR")
    parser.add_argument("--step_size", type=int, default=25)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    # arguments for saving
    parser.add_argument("--model_save_loc", type=str, default="weights/")
    parser.add_argument("--model_name", type=str)
    # arguments for loading
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--load_model_loc", type=str, default="weights/")
    parser.add_argument("--load_model_name", type=str, default="")

    # num gpus, gpu id
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    # # check if model_save_loc exists
    # if args.do_train:

    #     pass
    test_train(args)
    

    # load dataset, datasets are torch datasets, features of type float32, and labels of type int64
    # train, val, test = load_dataset(name=args.dataset, stratify=args.stratify, \
    #                                 test_ratio= args.test_size, val_ratio=args.val_size, \
    #                                       random_state = args.seed)


    # # create dataloaders 
    # train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
    # val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
    # test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)

    # # create model
    # model = DeepLDA(n_hidden=args.n_hidden, dropout=args.dropout, out_size=args.output_size, input_size=train[0][0].shape[0])

    
        



    







    #### TESTING FUNCTIONS ####
    # test_parsing(args)
    # TODO: test dataloaders
    # test_dataloaders(train_loader, val_loader, test_loader)

