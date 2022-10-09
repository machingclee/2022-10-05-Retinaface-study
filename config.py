anchor_ratios = [0.5, 1, 2]
anchor_scales = [64, 256, 512]
num_anchors = len(anchor_ratios) * len(anchor_scales)
batch_size = 1
image_shape = (1024, 1024)
input_height, input_width = image_shape
img_shapes = [
    (input_height // 8, input_width // 8),
    (input_height // 16, input_width // 16),
    (input_height // 32, input_width // 32)
]
fpn_out_channels = 256
rpn_n_sample = 128
rpn_pos_ratio = 0.5
target_pos_iou_thres = 0.7
target_neg_iou_thres = 0.3

rpn_n_sample = 128
rpn_pos_ratio = 0.5
n_landmark_coordinates = 10
img_dir_train = "dataset/WIDER_train/images"
img_dir_val = "dataset/WIDER_val/images"
bar_format = "{desc}: {percentage:.1f}%|{bar:15}| {n}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"

training_annotation = "dataset/wider_face_annotation/train/label.txt"
validation_annotation = "dataset/wider_face_annotation/val/label.txt"
pred_score_thresh = 0.9

model_visualization_dir = "performance_check"
save_per_batches = 50

longest_side_length = 1024
final_nms_iou = 0.01
