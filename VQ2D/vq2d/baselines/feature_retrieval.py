from typing import Any, Dict, Sequence

import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from einops import rearrange, asnumpy

from ..structures import BBox
from .utils import extract_window_with_context


def perform_retrieval(
    clip_frames: Sequence[np.ndarray],
    visual_crop: Dict[str, Any],
    query_frame: int,
    net: DefaultPredictor,
    batch_size: int = 8,
    downscale_height: int = 700,
    recency_factor: float = 1.0,  # Search only within the most recent frames.
    subsampling_factor: float = 1.0,  # Search only withiin a subsampled set of frames.
):
    """
    Given a visual crop and frames from a clip, retrieve the bounding box proposal
    from each frame that is most similar to the visual crop.
    """
    print("At the feature retrieval")
    #print(asjad)
    
    
    vc_fno = visual_crop["frame_number"]
    owidth, oheight = visual_crop["original_width"], visual_crop["original_height"]

    # Load visual crop frame
    reference = clip_frames[vc_fno]  # RGB format
    ## Resize visual crop if stored aspect ratio was incorrect

    if (reference.shape[0] != oheight) or (reference.shape[1] != owidth):
        reference = cv2.resize(reference, (owidth, oheight))
    reference = torch.as_tensor(rearrange(reference, "h w c -> () c h w"))
    reference = reference.float()
    ref_bbox = (
        int(visual_crop["x"]),
        int(visual_crop["y"]),
        int(visual_crop["x"]) + int(visual_crop["width"]),
        int(visual_crop["y"]) + int(visual_crop["height"]),
    )
    reference = extract_window_with_context(
        reference,
        ref_bbox,
        net.cfg.INPUT.REFERENCE_CONTEXT_PAD,
        size=net.cfg.INPUT.REFERENCE_SIZE,
        pad_value=125,
    )
    reference = rearrange(asnumpy(reference.byte()), "() c h w -> h w c")
    # Define search window
    search_window = list(range(0, query_frame))
    ## Pick recent k% of frames
    window_size = int(round(len(search_window) * recency_factor))
    if len(search_window[-window_size:]) > 0:
        search_window = search_window[-window_size:]
    ## Subsample only k% of frames
    window_size = len(search_window)
    idxs_to_sample = np.linspace(
        0, window_size - 1, int(subsampling_factor * window_size)
    ).astype(int)
    if len(idxs_to_sample) > 0:
        search_window = [search_window[i] for i in idxs_to_sample]

    # Load reference frames and perform detection
    ret_bboxes = []
    ret_scores = []
    ret_imgs = []
    # Batch extract predictions
    #print(batch_size)
    #print(len(search_window))
    for i in range(0, len(search_window), batch_size):
        bimages = []
        breferences = []
        image_scales = []
        i_end = min(i + batch_size, len(search_window))
        for j in range(i, i_end):
            image = clip_frames[search_window[j]]  # RGB format
            if image.shape[:2] != (oheight, owidth):
                image = cv2.resize(image, (owidth, oheight))
                print("Incorrect aspect ratio encountered!")
            # Scale-down image to reduce memory consumption
            image_scale = float(downscale_height) / image.shape[0]
            image = cv2.resize(image, None, fx=image_scale, fy=image_scale)
            bimages.append(image)
            breferences.append(reference)
            ret_imgs.append(clip_frames[search_window[j]].copy())
            image_scales.append(image_scale)
        # Perform inference
        #print("performing inference")
        all_outputs = net(bimages, breferences)
        #print("done with inference")
        # Unpack outputs
        for j in range(i, i_end):
            instances = all_outputs[j - i]["instances"]
            image_scale = image_scales[j - i]
            # Re-scale bboxes
            ret_bbs = (
                asnumpy(instances.pred_boxes.tensor / image_scale).astype(int).tolist()
            )
            ret_bbs = [BBox(search_window[j], *bbox) for bbox in ret_bbs]
            ret_scs = asnumpy(instances.scores).tolist()
            ret_bboxes.append(ret_bbs)
            ret_scores.append(ret_scs)

            #print(len(ret_bbs))
            #print(instances.pred_boxes.tensor[0]/image_scale)

            #print(ret_scs)

            if False :
                
                img = bimages[j-i]
                for k in range(len(ret_bbs)):
                    x1 = ret_bbs[k].x1
                    x2 = ret_bbs[k].x2
                    y1 = ret_bbs[k].y1
                    y2 = ret_bbs[k].y2

                    #print(len(img[j-i][0][0]))
                    
                    if round(ret_scs[k],3) > 0.01:
                        img = cv2.rectangle(bimages[j-i], (y1, x1), (y2, x2), (255,0,0), 2)
                        img = cv2.putText(img, str(round(ret_scs[k],3)), (y1, x1),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (144,238,144), 1)


                #img = bimages[i]
                #print(ret_bbs)
                cv2.imwrite("/home/asjad.s/results/all_frames/"+str(j)+".JPG",img)
                #cv2.imwrite("/home/asjad.s/nodes/r"+str(j)+".JPG",breferences[j-i])
                del img

        del all_outputs
    #print("WAS IN FEATURE RETRIEAL.PY")
    
    
    return ret_bboxes, ret_scores, ret_imgs, reference
