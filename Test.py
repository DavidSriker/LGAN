import argparse
from Model import *
from sklearn.model_selection import train_test_split
import random
import torchvision
from data_utils.DataProcess import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=75, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="prostate", help="name of the dataset")
    parser.add_argument("--model_name", type=str, default="lgan", help="name of the model")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    opt = parser.parse_args()
    print(opt)

    assert opt.dataset_name in ['lung', 'prostate'], print("dataset can be either (lung, prostate)")

    if opt.dataset_name == 'lung':
        image_shape = (256, 256)
        transformations_test = transforms.Compose([transforms.Resize((256, 256)),
                                                    transforms.ToTensor()])
        lung_path = os.path.join("data", "Lung_Segmentation")
        if os.path.exists(os.path.join(lung_path, "split")):
            test_img = pd.read_pickle(os.path.join(lung_path, "split", "test_img.pkl"))
            test_seg = pd.read_pickle(os.path.join(lung_path, "split", "test_seg.pkl"))
        else:
            print("You should run the Train.py before the Test.py")
            exit()

        test_set = LungSeg(test_img, test_seg, transforms=transformations_test)
    elif opt.dataset_name == 'prostate':
        image_shape = (256, 256)
        transformations_train = transforms.Compose([transforms.ToPILImage(mode="L"),
                                                    transforms.ToTensor()])
        transformations_val = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((256, 256)),
                                                  transforms.ToTensor()])

        prostate_path = os.path.join("data", "Prostate_Segmentation")
        if os.path.exists(os.path.join(prostate_path, "split")):
            test_img = np.load(os.path.join(prostate_path, "split", "test_img.npy"))
            test_seg = np.load(os.path.join(prostate_path, "split", "test_seg.npy"))
        else:
            print("You should run the Train.py before the Test.py")
            exit()
        test_set = ProstateSeg(test_img, test_seg, transformations_train)

    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)
    lgan = TesterLGAN((1, *image_shape), opt)
    lgan.test(test_loader, opt.dataset_name)
    print("Done Testing!")