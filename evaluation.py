import numpy as np
from utils import iou, CircleParams


def check_iou(model, data, device, thres=0.5, bs=10):
    correct = 0
    for inputs, labels in data:
        model.eval()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs.float())
        labels = labels
        for i, j in zip(outputs, labels):
            x_p, y_p, r_p = i.detach().cpu().numpy() * [100, 100, 49]
            x_p, y_p, r_p = int(x_p), int(y_p), int(r_p)
            x_l, y_l, r_l = j.cpu().numpy() * [100, 100, 49]
            x_l, y_l, r_l = int(x_l), int(y_l), int(r_l)
            current_iou = iou(CircleParams(x_l, y_l, r_l), CircleParams(x_p, y_p, r_p))
            if current_iou > thres:
                correct += 1
    return correct / (len(data) * bs)
