import argparse
from Model import *
from sklearn.model_selection import train_test_split
import random
import torchvision
from data_utils.DataProcess import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="lung", help="name of the dataset")
    parser.add_argument("--model_name", type=str, default="lgan", help="name of the model")
    parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving generator outputs")
    opt = parser.parse_args()
    print(opt)

    assert opt.dataset_name in ['lung', 'prostate'], print("dataset can be either (lung, prostate)")

    if opt.dataset_name == 'lung':
        image_shape = (256, 256)
        transformations_train = transforms.Compose([transforms.Resize((256, 256)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor()])
        transformations_val = transforms.Compose([transforms.Resize((256, 256)),
                                                  transforms.ToTensor()])

        lung_path = os.path.join("data", "Lung_Segmentation")
        if os.path.exists(os.path.join(lung_path, "split")):
            train_img = pd.read_pickle(os.path.join(lung_path, "split", "train_img.pkl"))
            val_img = pd.read_pickle(os.path.join(lung_path, "split", "val_img.pkl"))
            train_seg = pd.read_pickle(os.path.join(lung_path, "split", "train_seg.pkl"))
            val_seg = pd.read_pickle(os.path.join(lung_path, "split", "val_seg.pkl"))
        else:
            img_df, seg_df = lungDataProcess()
            train_img, test_img, train_seg, test_seg = train_test_split(img_df, seg_df, test_size=0.02)
            train_img, val_img, train_seg, val_seg = train_test_split(train_img, train_seg, test_size=0.01)

            os.mkdir(os.path.join(lung_path, "split"))
            train_img.to_pickle(os.path.join(lung_path, "split", "train_img.pkl"))
            val_img.to_pickle(os.path.join(lung_path, "split", "val_img.pkl"))
            train_seg.to_pickle(os.path.join(lung_path, "split", "train_seg.pkl"))
            val_seg.to_pickle(os.path.join(lung_path, "split", "val_seg.pkl"))
            test_img.to_pickle(os.path.join(lung_path, "split", "test_img.pkl"))
            test_seg.to_pickle(os.path.join(lung_path, "split", "test_seg.pkl"))

        train_set = LungSeg(train_img, train_seg, transforms=transformations_train)
        val_set = LungSeg(val_img, val_seg, transforms=transformations_val)

    elif opt.dataset_name == 'prostate':
        image_shape = (256, 256)
        transformations_train = transforms.Compose([transforms.ToPILImage(mode="L"),
                                                    transforms.ToTensor()])
        transformations_val = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((256, 256)),
                                                  transforms.ToTensor()])

        prostate_path = os.path.join("data", "Prostate_Segmentation")
        if os.path.exists(os.path.join(prostate_path, "split")):
            train_img = np.load(os.path.join(prostate_path, "split", "train_img.npy"))
            train_seg = np.load(os.path.join(prostate_path, "split", "train_seg.npy"))
            val_img = np.load(os.path.join(prostate_path, "split", "val_img.npy"))
            val_seg = np.load(os.path.join(prostate_path, "split", "val_seg.npy"))

        else:
            img_np, seg_np = prostateDataProcess()
            train_img, test_img, train_seg, test_seg = train_test_split(img_np, seg_np, test_size=0.02)
            train_img, val_img, train_seg, val_seg = train_test_split(train_img, train_seg, test_size=0.01)

            os.mkdir(os.path.join(prostate_path, "split"))
            np.save(os.path.join(prostate_path, "split", "train_img"), train_img)
            np.save(os.path.join(prostate_path, "split", "train_seg"), train_seg)
            np.save(os.path.join(prostate_path, "split", "val_img"), val_img)
            np.save(os.path.join(prostate_path, "split", "val_seg"), val_seg)
            np.save(os.path.join(prostate_path, "split", "test_img"), test_img)
            np.save(os.path.join(prostate_path, "split", "test_seg"), test_seg)

        train_set = ProstateSeg(train_img, train_seg, transformations_train)
        val_set = ProstateSeg(val_img, val_seg, transformations_val)

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    lgan = TrainerLGAN((1, *image_shape), opt)
    lgan.train(train_loader, val_loader, data_name=opt.dataset_name)

    print("Done Training!")
