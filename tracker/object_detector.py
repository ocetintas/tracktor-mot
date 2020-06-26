import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes

from collections import OrderedDict


class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes, nms_thresh=0.5):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        self.roi_heads.nms_thresh = nms_thresh
        self.backbone_features = None
        self.im_size = None
        self.im_transformed_size = None

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

    def cache(self, img):
        """
        Cache the current frame dimension and backbone features for bbox regression
        """
        device = list(self.parameters())[0].device
        img = img.to(device)

        # Cache dimensions of the input and after GeneralizedRCNNTransform
        self.im_size = img.shape[-2:]
        im_transformed, targets = self.transform(img)
        self.im_transformed_size = im_transformed.image_sizes[0]

        # Cache features as suggested by GeneralizedRCNN class of torchivison.models.detection
        self.backbone_features = self.backbone(im_transformed.tensors)
        if isinstance(self.backbone_features, torch.Tensor):
            self.backbone_features = OrderedDict([('0', self.backbone_features)])

    def bbox_regression(self, boxes):
        """
        Tracking of the objects from previous frame happens here. Bbox regressor of the FRCNN_FPN is used as the
        tracking mechanism
        """
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        # Resize to im_transformed
        boxes = resize_boxes(boxes, self.im_size, self.im_transformed_size)

        # Forward pass of the RoIHeads adapted from the implementation of roi_heads.py of torchvision.models.detection
        box_features = self.roi_heads.box_roi_pool(self.backbone_features, [boxes], [self.im_transformed_size])
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)
        pred_boxes = self.roi_heads.box_coder.decode(box_regression, [boxes])
        pred_scores = F.softmax(class_logits, -1)

        # Remove predictions with the background label
        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1)
        pred_scores = pred_scores[:, 1:].squeeze(dim=1)
        # Resize to im
        pred_boxes = resize_boxes(pred_boxes, self.im_transformed_size, self.im_size)

        return pred_boxes.detach(), pred_scores.detach()



