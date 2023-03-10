import argparse

class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--training_data_path', default='/media/wisccitl/han/KITTI_new/dataset/train',type=str, help="training data file path")
        self.parser.add_argument('--testing_data_path', type=str, default="F:\\KITTI_new\\dataset\\val",help="testing data file path")
        self.parser.add_argument('--validating_data_path', type=str, default="F:\\KITTI_new\\dataset\\val",help="validating data file path")
        self.parser.add_argument("--image_height", type=int, default=256, help="size of image height") #720
        self.parser.add_argument("--image_width", type=int, default=256, help="size of image width") #1280
        self.parser.add_argument("--channels", type=int, default=3, help="number of image channels")
        self.parser.add_argument("--batch_size", type=int, default=1, help="batch size of training process")
        self.parser.add_argument("--num_classes", type=int, default=12, help="the number of classes")
        self.parser.add_argument("--base_lr", type=float, default=0.001, help="learning rate")
        self.parser.add_argument("--max_epochs", type=int, default=100, help="maximum epoch")
        self.parser.add_argument("--max_iterations", type=int, default=100000, help="maximun iteration")
        self.parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
        self.parser.add_argument("--seed", type=int, default=42, help="random seed for cuda")
        self.parser.add_argument("--save_interval", type=int, default=10, help="the model weights saving interval")
        self.parser.add_argument("--model_path", type=str, default="model/", help="the path of saving logs")
        self.parser.add_argument("--model_weight_path", type=str, default="model\\CNN+TNT\\best.pth", help="the path of saving weights")
        self.parser.add_argument("--train", type=bool, default=None, help="true to train")
        self.parser.add_argument("--test", type=bool, default=None, help="true to test")
        self.parser.add_argument("--continue_training", type=bool, default=False, help="whether to continue from the last checkpoint")
        self.parser.add_argument("--ignore_background_class", type=bool, default=False, help="whether mIoU considered background class (0)")
        self.parser.add_argument("--results_save_path", type=str, default="results", help="The folder path for saving the predictions")
        self.opt = self.parser.parse_args()
