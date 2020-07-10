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
        self.im_size = None

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

    def bbox_regression(self, img, boxes):
        """
        Tracking of the objects from previous frame with the bounding box regressor of the FRCNN_FPN
        """

        # Move image and boxes to the device
        device = list(self.parameters())[0].device
        img = img.to(device)
        boxes = boxes.to(device)

        # Perform input transformation before feeding the image into a GeneralizedRCNN model of torchvision
        img_size = img.shape[-2:]
        img_transformed, targets = self.transform(img)
        img_transformed_size = img_transformed.image_sizes[0]

        # Calculate the backbone features and put them in a compatible format with RoIHeads and RPN classes of torchvision
        backbone_features = self.backbone(img_transformed.tensors)
        if isinstance(backbone_features, torch.Tensor):
            backbone_features = OrderedDict([('0', backbone_features)])

        # Resize boxes to img_transformed size
        boxes = resize_boxes(boxes, img_size, img_transformed_size)

        # Forward pass of the RoIHeads class of torchvision
        box_features = self.roi_heads.box_roi_pool(backbone_features, [boxes], [img_transformed_size])
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        # Post-process the detections
        boxes = self.roi_heads.box_coder.decode(box_regression, [boxes])
        scores = F.softmax(class_logits, -1)

        # Remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]

        # Put the tensors in the correct shape for the Track class
        boxes = boxes.squeeze(dim=1)
        scores = scores.squeeze(dim=1)

        # Resize to img size
        boxes = resize_boxes(boxes, img_transformed_size, img_size)

        return boxes.detach().cpu(), scores.detach().cpu()



