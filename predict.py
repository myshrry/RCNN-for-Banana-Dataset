from torchvision import transforms
import torch
from selective_search import Selective_search
import utils
from torch import nn
from PIL import Image
import numpy as np


def predict(img_path, alexnet, linear_net):
    proposal_region = Selective_search.cal_pro_region(img_path)
    img = Image.open(img_path).convert('RGB')
    img_tensor = transforms.functional.to_tensor(img)

    transform = transforms.Compose([transforms.Resize((227, 227)),
                                    transforms.Normalize(mean=[0.564, 0.529, 0.385], std=[1, 1, 1])])
    locs, offset = [], []
    for loc in proposal_region:
        with torch.no_grad():
            crop_region = img_tensor[:, loc[1]:loc[3], loc[0]:loc[2]]
            crop = transform(crop_region).unsqueeze(0)
            if torch.argmax(alexnet(crop)).item():
                features = alexnet.features(crop)
                offset.append(linear_net(features).squeeze(0))
                locs.append(torch.tensor(loc, dtype=torch.float32))
    if locs is not None:
        offset, locs = torch.vstack(offset), torch.vstack(locs)
        index = offset.abs().sum(dim=1).argmin().item()
        result = locs[index] + offset[index]
        utils.draw_box(img, np.array(result.unsqueeze(0)))
    else:
        utils.draw_box(img)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# =====================加载Alexnet训练参数====================
alexnet_state_dict = './model/classify_5th_model.pth'
alexnet = utils.get_Alexnet()
alexnet.load_state_dict(torch.load(alexnet_state_dict, map_location=device))
alexnet.to(device)
alexnet.eval()

# =====================加载linear_net训练参数==================
linear_net = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), nn.Linear(256*6*6, 4))
linear_state_dict = './model/regression_15th_model.pth'
linear_net.load_state_dict(torch.load(linear_state_dict, map_location=device))
linear_net.to(device)
linear_net.eval()

if __name__=='__main__':
    img_path = './test_imgs/73.png'
    predict(img_path, alexnet, linear_net)

