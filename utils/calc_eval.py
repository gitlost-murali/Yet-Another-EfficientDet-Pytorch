"""
COCO-Style Evaluations

put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import torch
import yaml

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import invert_affine, postprocess

def evaluate_mAP(imgs, imgs_ids, framed_metas, regressions, classifications, anchors, threshold=0.05,nms_threshold=0.5):
    results = [] # This is used for storing evaluation results.
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    preds = postprocess(imgs,
                    torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
                    regressBoxes, clipBoxes,
                    threshold, nms_threshold)

    if not preds:
        return

    preds = invert_affine(framed_metas, preds)
    for i in range(len(preds)):
        scores = preds[i]['scores']
        class_ids = preds[i]['class_ids']
        rois = preds[i]['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                if score < threshold:
                    break
                image_result = {
                    'image_id': imgs_ids[i],
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)
    return results

def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def calc_mAP_fin(project_name='shape',
                 set_name='val',
                 evaluation_pred_file='datasets/shape/predictions/instances_bbox_results.json'):
    val_gt = f'datasets/{project_name}/annotations/instances_{set_name}.json'
    max_images = 10000
    coco_gt = COCO(val_gt)
    image_ids = coco_gt.getImgIds()[:max_images]
    _eval(coco_gt, image_ids, evaluation_pred_file)
