import torch
import torch.utils.data
import torch.optim

from torchvision import datasets, transforms
from torchvision.models.efficientnet import efficientnet_v2_s

from tqdm import tqdm

import numpy as np


device = torch.device('cuda:1')


def custom_efficientnet():
    model = efficientnet_v2_s(pretrained=True).to(device)
    model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features, out_features=1, bias=True).to(device)
    print(model)

    return model



model = custom_efficientnet()


train_data = datasets.ImageFolder('./small-frame-chunks', transform=transforms.ToTensor())

generator = torch.Generator().manual_seed(42)
# split = int(0.8 * len(train_data))
# train_data, test_data = torch.utils.data.random_split(
#     dataset=train_data,
#     lengths=[split, len(train_data) - split],
#     generator=generator
# )


train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=True)



with open('model.pth', 'rb') as f:
    model = torch.load(f)


threshold = 0.5
thresholds = [*enumerate([0.01, 0.02, 0.04, 0.08, 0.16])]
print(thresholds)


#validation doesnt requires gradient
with torch.no_grad():
    model.eval()

    misclassifieds = []
    counts = []
    idx = 0
    tp = [0] * len(thresholds)
    tn = [0] * len(thresholds)
    fp = [0] * len(thresholds)
    fn = [0] * len(thresholds)
    for x_batch, y_batch in tqdm(train_loader):
        idx += 1
        print(idx, '/', len(train_loader))

        x_batch = x_batch.to(device)
        y_batch = y_batch.unsqueeze(1)#.float() #convert target to same nn output shape
        # y_batch = y_batch.to(device)

        #model to eval mode
        model.eval()

        yhat = model(x_batch)
        res = yhat > threshold
        res = res.int()

        for i, threshold in thresholds:
            res = yhat > threshold
            y_ = res.int().squeeze().detach().cpu().numpy()
            y = y_batch.int().squeeze().detach().cpu().numpy()

            for _y_, _y in zip(y_, y):
                if _y_ == 1 and _y == 1:
                    tp[i] += 1
                elif _y_ == 0 and _y == 0:
                    tn[i] += 1
                elif _y_ == 1 and _y == 0:
                    fp[i] += 1
                elif _y_ == 0 and _y == 1:
                    fn[i] += 1

        misclassified = (y_batch.to(device).int() != res).sum()
        count = len(y_batch)

        misclassifieds.append(int(misclassified))
        counts.append(count)
    
    print(misclassifieds)
    print(counts)

    print((1 - (sum(misclassifieds) / sum(counts)) * 100))


    for i, threshold in thresholds:
        print('Threshold:', threshold, '------------------------------------------------------------')
        print('TP:', tp[i])
        print('TN:', tn[i])
        print('FP:', fp[i])
        print('FN:', fn[i])

        print('Accuracy:', (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i]))
        print('Precision:', tp[i] / (tp[i] + fp[i]))
        print('----------- Recall:', tp[i] / (tp[i] + fn[i]))
        print('F1:', 2 * (tp[i] / (tp[i] + fp[i])) * (tp[i] / (tp[i] + fn[i])) / ((tp[i] / (tp[i] + fp[i])) + (tp[i] / (tp[i] + fn[i]))))
        print('Specificity:', tn[i] / (tn[i] + fp[i]))
        print('Sensitivity:', tp[i] / (tp[i] + fn[i]))
        print('MCC:', (tp[i] * tn[i] - fp[i] * fn[i]) / np.sqrt((tp[i] + fp[i]) * (tp[i] + fn[i]) * (tn[i] + fp[i]) * (tn[i] + fn[i])))
        print('FPR:', fp[i] / (fp[i] + tn[i]))
        print('----------- FNR:', fn[i] / (fn[i] + tp[i]))
        print('----------------,',  (tp[i] / (tp[i] + fn[i])) + (fn[i] / (fn[i] + tp[i])))
        print('NPV:', tn[i] / (tn[i] + fn[i]))
        print('FDR:', fp[i] / (fp[i] + tp[i]))
        print('FOR:', fn[i] / (fn[i] + tn[i]))
        print('ACC:', (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i]))