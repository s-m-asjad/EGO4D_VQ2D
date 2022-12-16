from typing import Dict, Tuple, Any

import torch
"""custom imports""" 
import timm
import cv2
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
from datetime import datetime
import time
import numpy
"""end"""

from detectron2.config import CfgNode, configurable
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage

__all__ = ["SiameseRCNN"]


@META_ARCH_REGISTRY.register()
class SiameseRCNN(GeneralizedRCNN):
    """
    Siamese R-CNN. Any model that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    In addition to GeneralizedRCNN, this takes in a reference crop as input,
    extracts features from it, and passes it to the ROIHead.
    """

    @configurable
    def __init__(self, *args, **kwargs) -> None:
        

        super().__init__(*args, **kwargs)
        
        self.object_classifier = timm.create_model('beit_large_patch16_224_in22k', pretrained=True)
        self.object_classifier.eval()
        self.object_classifier.cuda()

    @classmethod
    def from_config(cls, cfg: CfgNode) -> Dict[str, Any]:
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(
                cfg, backbone.output_shape()
            ),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * reference: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images, references = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        ref_features = self.backbone(references.tensor)
        ref_features = {k: v.view(-1, 1, *v.shape[1:]) for k, v in ref_features.items()}

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, features, proposals, ref_features, gt_instances
        )
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, Any]],
        do_postprocess: bool = True,
    ) -> Any:
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        #print(list(batched_inputs))
        #height = 700 width = 1244

        images, references = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        ref_features = self.backbone(references.tensor)
        ref_features = {k: v.view(-1, 1, *v.shape[1:]) for k, v in ref_features.items()}

        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        
        empty_index=[]
        
        start_time = time.time()
        transform = T.ToPILImage()#T.Resize((224,224),interpolation=T.InterpolationMode.BICUBIC)
        if True:

            sub_batch_size = 1

            loop_indices = []


            if sub_batch_size == 1:
                loop_indices = [L for L in range(len(proposals))]
            elif sub_batch_size == 2:
                loop_indices = [L*2 for L in range(int(  (len(proposals) + 1 ) /2)) ]
            elif sub_batch_size == 3:
                loop_indices = [L*3+1 for L in range(  int( (len(proposals) + 2 ) /3))]
                if (len(proposals)%3) == 1:
                    loop_indices[len(loop_indices)-1] = loop_indices[len(loop_indices)-1]-1




            top_classes_count = 5
            
        


            #print(loop_indices)
            for p in loop_indices:
                retain_index = []
                
                #print(p)


                #retain_index = [idx for idx in range(len(proposals[p].get_fields()["proposal_boxes"])) if 5389 in torch.topk(self.object_classifier(torch.cat([TF.to_tensor(transform(images[p][: , int(b[1]):int(b[3]) , int(b[0]):int(b[2])]).resize((224, 224), Image.ANTIALIAS)).unsqueeze(0) for b in [proposals[p].get_fields()["proposal_boxes"].tensor[prop_box] for prop_box in range(len(proposals[p].get_fields()["proposal_boxes"]))]]).cuda(non_blocking=True)),top_classes_count)[1].tolist()[idx] ]     #idx in range(len())

                
                with torch.no_grad():
                    classification = torch.topk(self.object_classifier(torch.cat([TF.to_tensor(transform(images[p][: , int(b[1]):int(b[3]) , int(b[0]):int(b[2])]).resize((224, 224), Image.ANTIALIAS)).unsqueeze(0) for b in [proposals[p].get_fields()["proposal_boxes"].tensor[prop_box] for prop_box in range(len(proposals[p].get_fields()["proposal_boxes"]))]]).cuda(non_blocking=True)),top_classes_count)[1].tolist()

                


                #classification = torch.topk(self.object_classifier(torch.cat([transform(images[p][: , int(b[1]):int(b[3]) , int(b[0]):int(b[2])]).unsqueeze(0) for b in [proposals[p].get_fields()["proposal_boxes"].tensor[prop_box] for prop_box in range(len(proposals[p].get_fields()["proposal_boxes"]))]]).cuda(non_blocking=True)),top_classes_count)[1].tolist()
                
                
                for idx in range(len(classification)):
                    
                    #if 4773 in classification[idx] or 4780 in classification[idx] or 5402 in classification[idx] :
                    lbls = [3682, 5389, 7690, 10791, 5170, 7688, 6924, 5036, 11828, 8037, 6496]
                    ctr=0
                    while ctr<1:
                        if lbls[ctr] in classification[idx]:
                            retain_index.append(idx)
                            break
                        ctr=ctr+1                       
                        
                              

                #USING ONLY THE CLASS OF QUERY IMAGE
                if len(retain_index)>0:
                    #Selecting only the proposal boxes where classification was successful

                    #proposals[p].get_fields()["proposal_boxes"].tensor=proposals[p].get_fields()["proposal_boxes"].tensor[retain_index]
                    #proposals[p].get_fields()["objectness_logits"].tensor=proposals[p].get_fields()["objectness_logits"][retain_index]

                    prop_boxes_final = proposals[p].get_fields()["proposal_boxes"].tensor[retain_index]
                    logits_final =proposals[p].get_fields()["objectness_logits"][retain_index]

                    proposals[p].get_fields()["proposal_boxes"].tensor = prop_boxes_final.detach().clone()
                    proposals[p].get_fields()["objectness_logits"].tensor = logits_final.detach().clone()



                    if sub_batch_size == 2 and p+1<len(proposals):

                        proposals[p+1].get_fields()["proposal_boxes"].tensor = prop_boxes_final.detach().clone()
                        proposals[p+1].get_fields()["objectness_logits"].tensor = logits_final.detach().clone()

                    elif sub_batch_size == 3 and len(proposals)%3 != 1:

                        proposals[p-1].get_fields()["proposal_boxes"].tensor = prop_boxes_final.detach().clone()
                        proposals[p-1].get_fields()["objectness_logits"].tensor = logits_final.detach().clone()

                        if len(proposals)%3 == 0:

                            proposals[p+1].get_fields()["proposal_boxes"].tensor = prop_boxes_final.detach().clone()
                            proposals[p+1].get_fields()["objectness_logits"].tensor = logits_final.detach().clone()


                    # IF NO OBJECT HAS THE CLASS OF THE QUERY IMAGE STORE THIS INDEX LOCATION AND CHANGE SCORE TO 0 AFTER SIAMESE PREDICTIONS
                else:
                    empty_index.append(p)

                    if sub_batch_size == 2 and p+1<len(proposals):
                        empty_index.append(p+1)
                    
                    elif sub_batch_size == 3 and len(proposals)%3 != 1:
                        empty_index.append(p-1)
                        if len(proposals)%3 == 0:
                            empty_index.append(p+1)
                    
          

        
        print("batch completed")
        print(time.time() - start_time)

        
        results, _ = self.roi_heads(images, features, proposals, ref_features, None)

        #print(results[0].get_fields()["pred_boxes"])
        
        #for ri in range(len(results)):

        # classification = [torch.topk(self.object_classifier(torch.cat([TF.to_tensor(transform(images[p][: , int(b[1]):int(b[3]) , int(b[0]):int(b[2])]).resize((224, 224), Image.ANTIALIAS)).unsqueeze(0) for b in [results[p].get_fields()["pred_boxes"].tensor[prop_box] for prop_box in range(len(results[p].get_fields()["pred_boxes"]))]]).cuda(non_blocking=True)),5)[1].tolist() for p in range(len(images)) ]

        # for cix in range(len(classification)):
        #     for idx in range(len(classification[cix])):                    
        #             #if 4773 in classification[idx] or 4780 in classification[idx] or 5402 in classification[idx] :
        #             lbls = [5389, 7690, 10791, 5170, 7688, 6924, 5036, 11828, 8037, 6496]
        #             ctr=0
        #             if lbls[ctr] in classification[cix][idx]:
        #                 pass
        #             else:
        #                 results[cix].get_fields()["scores"][idx] = 0



        
        for ei in range(len(empty_index)):
            idx = empty_index[ei]
            results[idx].get_fields()["scores"] = torch.zeros(len(results[idx].get_fields()["scores"]))

        
        




        

        if do_postprocess:
            assert (
                not torch.jit.is_scripting()
            ), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            )
        else:
            return results

    def preprocess_image(
        self, batched_inputs: Tuple[Dict[str, torch.Tensor]]
    ) -> Tuple[ImageList, ImageList]:
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        references = [x["reference"].to(self.device) for x in batched_inputs]
        references = [(x - self.pixel_mean) / self.pixel_std for x in references]
        references = ImageList.from_tensors(references, 0)
        return images, references
