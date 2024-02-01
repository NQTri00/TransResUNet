
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from model import TResUnet
from utils import create_dir, seeding
from utils import calculate_metrics
from train import load_data, load_test_data


def evaluate(model, save_path, test_x, test_y, size):
    """ Loading other comparitive model masks """
    comparison_path = "/Kvasir-SEG/"

    unet_mask = sorted(glob(os.path.join(comparison_path, "UNET", "results", "Kvasir-SEG", "mask", "*")))
    deeplabv3plus_mask = sorted(glob(os.path.join(comparison_path, "DeepLabV3+_50", "results", "Kvasir-SEG", "mask", "*")))


    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()
            heatmap, y_pred = model(image, heatmap=True)
            y_pred = torch.sigmoid(y_pred)
            end_time = time.time() - start_time
            time_taken.append(end_time)

            """ Evaluation metrics """
            score = calculate_metrics(mask, y_pred)
            metrics_score = list(map(add, metrics_score, score))

            """ Predicted Mask """
            y_pred = y_pred[0].cpu().numpy()
            y_pred = np.squeeze(y_pred, axis=0)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype(np.int32)
            y_pred = y_pred * 255
            y_pred = np.array(y_pred, dtype=np.uint8)
            y_pred = np.expand_dims(y_pred, axis=-1)
            y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)

        """ Save the image - mask - pred """
        line = np.ones((size[0], 10, 3)) * 255
        cat_images = np.concatenate([
            save_img, line,
            save_mask, line,
            cv2.imread(unet_mask[i], cv2.IMREAD_COLOR), line,
            cv2.imread(deeplabv3plus_mask[i], cv2.IMREAD_COLOR), line,
            y_pred, line,
            heatmap], axis=1)

        cv2.imwrite(f"{save_path}/joint/{name}.jpg", cat_images)
        cv2.imwrite(f"{save_path}/mask/{name}.jpg", y_pred)
        cv2.imwrite(f"{save_path}/heatmap/{name}.jpg", heatmap)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    f2 = metrics_score[5]/len(test_x)

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TResUnet()
    model = model.to(device)
    checkpoint_path = "/content/drive/MyDrive/bkai/checkpoint-BKAI-IGH.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    path = "/Kvasir-SEG"
    # (train_x, train_y), (test_x, test_y) = load_data(path)
    # test_x = load_test_data(path)
    test_path = '/content/drive/MyDrive/bkai/test/test/'
    i = 1
    for file in os.listdir(test_path):
        img = test_path + file
        image = cv2.imread(img, cv2.IMREAD_COLOR)
        size = (256, 256)
        image = cv2.resize(image, size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)
        # mask = cv2.imread('/content/drive/MyDrive/bkai/train_gt/train_gt/0081835cf877e004e8bfb905b78a9139.jpeg')
        _, y_pred = model(image, heatmap=True)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred[0].cpu().detach().numpy()
        y_pred = np.squeeze(y_pred, axis=0)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)
        y_pred = y_pred * 255
        y_pred = np.array(y_pred, dtype=np.uint8)
        y_pred = np.expand_dims(y_pred, axis=-1)
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)

        image = cv2.imread(img, cv2.IMREAD_COLOR)
        y_pred = cv2.resize(y_pred, (image.shape[1],image.shape[0]))
        cv2.imwrite(f"/content/drive/MyDrive/bkai/result/{file}", y_pred)
        print(i)
        i+=1
