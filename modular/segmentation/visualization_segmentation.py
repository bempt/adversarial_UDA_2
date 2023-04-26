import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torchvision import transforms as T
from tqdm.notebook import tqdm

import modular.segmentation.metrics_segmentation as metrics_segmentation

def show_pred_mask(image, mask, pred_mask, score, figsize=(20,10)):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=figsize)
    ax1.imshow(image)
    ax1.set_title('Picture');

    ax2.imshow(mask)
    ax2.set_title('Ground truth')
    ax2.set_axis_off()

    ax3.imshow(pred_mask)
    ax3.set_title('UNet-ResNet50 | mIoU {:.3f}'.format(score))
    ax3.set_axis_off()

def inference(model, image):
    start_time = time.time()
    output = model(image)
    end_time = time.time()
    inference_time = end_time - start_time
    return output, inference_time

def predict_image_mask_miou(device, model, image, mask, mean, std):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        output, inference_time = inference(model, image)
        score = metrics_segmentation.mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score, inference_time

def show_inference_time(inference_time):
    print('Inference time: {:.3f} seconds'.format(inference_time))

def miou_score(model, test_set, device, mean, std):
    score_iou = []
    # create variable for average inference time
    avg_inference_time = 0
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score, inference_time = predict_image_mask_miou(device, model, img, mask, mean, std)
        # add new inference time to average calculation
        avg_inference_time += inference_time
        score_iou.append(score)
    print('Test mIoU: {:.3f}'.format(np.mean(score_iou)))
    # calculate average inference time
    avg_inference_time /= len(test_set)
    show_inference_time(avg_inference_time)
    return score_iou

def predict_image_mask_pixel(device, model, image, mask, mean, std):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        # output = model(image)
        output, inference_time = inference(model, image)
        acc = metrics_segmentation.pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc, inference_time

def pixel_acc(model, test_set, device, mean, std):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc, inference_time = predict_image_mask_pixel(device, model, img, mask, mean, std)
        accuracy.append(acc)
    print('Test Accuracy: {:.3f}'.format(np.mean(accuracy)))
    return accuracy

def seg_visualization(test_set,
                      device,
                      model,
                      mean,
                      std,
                      instances=3):
    
    for i in range(instances):
        rand = np.random.randint(0, len(test_set))
        image, mask = test_set[rand]
        pred_mask, score, inference_time = predict_image_mask_miou(device, model, image, mask, mean, std)
        mob_miou = miou_score(model, test_set, device, mean, std)
        mob_acc = pixel_acc(model, test_set, device, mean, std)
        show_pred_mask(image, mask, pred_mask, score)