from torch.utils.tensorboard import SummaryWriter
import cv2
import os
import uuid
import numpy as np
import torchvision
import torch

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/camera_trap_experiment_1')

def display_tboard(preds, imgs, obj_list, imshow=False, imwrite=False,global_step=0,classification_loss=0,regression_loss=0):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])
        
    # create grid of images
    imgs = np.asarray(imgs)
    imgs = torch.from_numpy(imgs)   # (N, H, W, C)
    imgs.transpose_(1, 3)
    img_grid = torchvision.utils.make_grid(imgs)   # (N, C, H, W)
    # write to tensorboard
    writer.add_image('four_camtrap_images_with_predicted_bboxes', img_grid,global_step=global_step)

    print(classification_loss)
    # ...log the running loss
    writer.add_scalar('Classification loss',classification_loss,global_step=global_step)
    writer.add_scalar('Regression loss',regression_loss,global_step=global_step)
