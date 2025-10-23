from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch


class SeqTrackActor(BaseActor):
    """ Actor for training the SeqTrack"""

    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.BINS = cfg.MODEL.BINS
        self.seq_format = cfg.DATA.SEQ_FORMAT

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        outputs, target_seqs, gt_boxes_continuous = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(outputs, target_seqs, gt_boxes_continuous)

        return loss, status

    def forward_pass(self, data):
        n, b, _, _, _ = data['search_images'].shape  # n,b,c,h,w
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (n*b, c, h, w)
        search_list = search_img.split(b, dim=0)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])
        template_list = template_img.split(b, dim=0)
        feature_xz = self.net(images_list=template_list + search_list, mode='encoder')  # forward the encoder

        bins = self.BINS  # coorinate token
        start = bins + 1  # START token
        end = bins  # End token
        len_embedding = bins + 2  # number of embeddings, including the coordinate tokens and the special tokens

        # box of search region
        targets = data['search_anno'].permute(1, 0, 2).reshape(-1, data['search_anno'].shape[2])  # x0y0wh
        targets = box_xywh_to_xyxy(targets)  # x0y0wh --> x0y0x1y1
        targets = torch.max(targets, torch.tensor([0.]).to(targets))  # Truncate out-of-range values
        targets = torch.min(targets, torch.tensor([1.]).to(targets))

        # 🔥 Keep continuous boxes for IoU calculation
        gt_boxes_continuous = targets.clone()

        # different formats of sequence, for ablation study
        if self.seq_format != 'corner':
            targets = box_xyxy_to_cxcywh(targets)
            gt_boxes_continuous = box_xyxy_to_cxcywh(gt_boxes_continuous)

        box = (targets * (bins - 1)).int()  # discretize the coordinates

        if self.seq_format == 'whxy':
            box = box[:, [2, 3, 0, 1]]
            gt_boxes_continuous = gt_boxes_continuous[:, [2, 3, 0, 1]]

        batch = box.shape[0]
        # inpute sequence
        input_start = torch.ones([batch, 1]).to(box) * start
        input_seqs = torch.cat([input_start, box], dim=1)
        input_seqs = input_seqs.reshape(b, n, input_seqs.shape[-1])
        input_seqs = input_seqs.flatten(1)

        # target sequence
        target_end = torch.ones([batch, 1]).to(box) * end
        target_seqs = torch.cat([box, target_end], dim=1)
        target_seqs = target_seqs.reshape(b, n, target_seqs.shape[-1])
        target_seqs = target_seqs.flatten()
        target_seqs = target_seqs.type(dtype=torch.int64)

        outputs = self.net(xz=feature_xz, seq=input_seqs, mode="decoder")

        outputs = outputs[-1].reshape(-1, len_embedding)

        return outputs, target_seqs, gt_boxes_continuous

    def compute_losses(self, outputs, targets_seq, gt_boxes_continuous, return_status=True):
        # Get loss
        ce_loss = self.objective['ce'](outputs, targets_seq)
        # weighted sum
        loss = self.loss_weight['ce'] * ce_loss

        # 🔥 OPTIMIZATION: Compute IoU more efficiently without softmax
        with torch.no_grad():  # No need for gradients during IoU calculation
            # 🔥 OPTIMIZATION: Use argmax directly on logits instead of softmax + topk
            # This gives the same result (highest scoring bin) but much faster
            logits = outputs[:, :self.BINS]
            extra_seq = logits.argmax(dim=-1, keepdim=True)

            # Get predicted boxes (discretized)
            boxes_pred_discrete = extra_seq.reshape(-1, 5)[:, 0:-1]

            # Convert predicted boxes from discrete to continuous (0 to 1)
            boxes_pred_continuous = boxes_pred_discrete.float() / (self.BINS - 1)

            # Get ground truth boxes (already continuous)
            boxes_target_continuous = gt_boxes_continuous.reshape(-1, 4)

            # Reorder if needed (for whxy format)
            if self.seq_format == 'whxy':
                boxes_pred_continuous = boxes_pred_continuous[:, [2, 3, 0, 1]]

            # Convert to xyxy format for IoU calculation
            if self.seq_format != 'corner':
                boxes_pred_xyxy = box_cxcywh_to_xyxy(boxes_pred_continuous)
                boxes_target_xyxy = box_cxcywh_to_xyxy(boxes_target_continuous)
            else:
                boxes_pred_xyxy = boxes_pred_continuous
                boxes_target_xyxy = boxes_target_continuous

            # Calculate IoU
            try:
                # 🔥 IMPROVED: More robust IoU calculation
                iou_output = box_iou(boxes_pred_xyxy, boxes_target_xyxy)

                # Handle different return types from box_iou
                if isinstance(iou_output, (list, tuple)):
                    iou_matrix = iou_output[0]
                else:
                    iou_matrix = iou_output

                # Get diagonal (matching pairs) and compute mean
                if len(iou_matrix.shape) == 2 and iou_matrix.shape[0] == iou_matrix.shape[1]:
                    iou = iou_matrix.diag().mean()
                else:
                    iou = iou_matrix.mean()

            except Exception as e:
                # Fallback if box_iou has issues
                if self.settings.local_rank in [-1, 0]:
                    print(f"⚠️  Warning: IoU calculation failed: {e}")
                iou = torch.tensor(0.0).to(outputs.device)

        if return_status:
            # status for log
            status = {
                "Loss/total": loss.item(),
                "Loss/ce": ce_loss.item(),
                "IoU": iou.item()
            }
            return loss, status
        else:
            return loss

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)
        self.objective['ce'].to(device)