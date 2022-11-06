final_nms_iou = 0.3
rpn_n_sample = 128

input_img_size = 840

WIDER_TRAIN_LABEL_TXT = r"C:\Users\user\Repos\Python\2022-10-05-Retinaface-study\data\wider_face\wider_face_annotation\train\label.txt"
WIDER_VAL_LABEL_TXT = r"C:\Users\user\Repos\Python\2022-10-05-Retinaface-study\data\wider_face\wider_face_annotation\val\label.txt"
WIDER_TRAIN_IMG_DIR = r"C:\Users\user\Repos\Python\2022-10-05-Retinaface-study\data\wider_face\WIDER_train\images"
WIDER_VAL_IMG_DIR = r"C:\Users\user\Repos\Python\2022-10-05-Retinaface-study\data\wider_face\WIDER_val\images"
WIDER_N_LANDMARKS = 5

WFLW_TRAIN_LABEL_TXT = r"C:\Users\user\Repos\Python\2022-10-05-Retinaface-study\data\WFLW\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_train.txt"
WFLW_VAL_LABEL_TXT = r"C:\Users\user\Repos\Python\2022-10-05-Retinaface-study\data\WFLW\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_test.txt"
WFLW_TRAIN_IMG_DIR = r"C:\Users\user\Repos\Python\2022-10-05-Retinaface-study\data\WFLW\WFLW_images"
WFLW_VAL_IMG_DIR = r"C:\Users\user\Repos\Python\2022-10-05-Retinaface-study\data\WFLW\WFLW_images"


constrain_landmarks_prediction_into_bbox = False
model_visualization_dir = "performance_check"
visualize_result_per_batch = 10
pred_thres = 0.5
n_landmarks = 98

initial_lr = 1e-4
font_path = "fonts/wt014.ttf"

landm_dot_radius = 1
landm_numbering_font_size = 6

checkpoint = "weights/Resnet50_031.pth"
start_epoch = 32
bar_format = "{desc}: {percentage:.1f}%|{bar:15}| {n}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"

landm_loss_weight = 10
