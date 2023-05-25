import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from util.utils import mIOU
from dataset.fewshot import FewShot
import argparse
from torch.utils.data import DataLoader
from model.matching import MatchingNet
import os
import torch.nn.functional as F
import cv2
colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                (128, 64, 12)]
def parse_args():
    parser = argparse.ArgumentParser(description='Mining Latent Classes for Few-shot Segmentation')
    # basic arguments
    parser.add_argument('--data-root',
                        type=str,
                        required=True,
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='pascal',
                        choices=['pascal', 'coco', 'steel'],
                        help='training dataset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=4,
                        help='batch size of training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--crop-size',
                        type=int,
                        default=473,
                        help='cropping size of training samples')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')

    # few-shot training arguments
    parser.add_argument('--fold',
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3],
                        help='validation fold')
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    parser.add_argument('--episode',
                        type=int,
                        default=24000,
                        help='total episodes of training')
    parser.add_argument('--snapshot',
                        type=int,
                        default=1200,
                        help='save the model after each snapshot episodes')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed to generate tesing samples')
    parser.add_argument('--model_path',
                        type=str,
                        default="outdir/models/steel/fold_1/resnet50_1shot_51.32.pth",
                        help='random seed to generate tesing samples')
    parser.add_argument('--save_dir',
                    type=str,
                    default="outdir/models/steel/fold_1/pred_shot1_v8",
                    help='random seed to generate tesing samples')

    args = parser.parse_args()
    return args

def evaluate(model, dataloader, args):
    tbar = tqdm(dataloader)

    num_classes = 10

    # metric = mIOU(num_classes)

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, id_s_list, id_q) in enumerate(tbar):
        img_q, mask_q = img_q.cuda(), mask_q.cuda()
        orininal_h = img_q[0].shape[1]
        orininal_w = img_q[0].shape[2]
        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()
        cls = cls[0].item()

        base_dir = os.path.join(args.save_dir, id_q[0])
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        with torch.no_grad():

            pred, features, similarity_fg, similarity_bg = model(img_s_list, mask_s_list, img_q)
            pred = torch.argmax(pred, dim=1)


            # combine_sim = np.concatenate([similarity_bg[None, ...].cpu().numpy() * 10.0, similarity_fg[None, ...]].cpu().numpy() * 10.0, axis=0).argmax(axis=0).astype(np.uint8) * 255
            combine_sim = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
            # combine_sim = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1)
            combine_sim_temp = F.interpolate(combine_sim, size=(orininal_h, orininal_w), mode="bilinear", align_corners=True)
            combine_sim = (torch.argmax(combine_sim_temp, dim=1).squeeze(0).cpu().numpy() * 255).astype(np.uint8)



            for level, feature in features.items():
                feature_vis((orininal_h, orininal_w), base_dir, level, feature)
            
            similarity_fg = F.interpolate(similarity_fg[None, ...], size=(orininal_h, orininal_w), mode='bilinear', align_corners=True).squeeze(0).squeeze(0).cpu().numpy()
            similarity_bg = F.interpolate(similarity_bg[None, ...], size=(orininal_h, orininal_w), mode='bilinear', align_corners=True).squeeze(0).squeeze(0).cpu().numpy()

            combine_sim_after = np.concatenate([similarity_bg[None, ...], similarity_fg[None, ...]], axis=0).argmax(axis=0).astype(np.uint8) * 255  # 如果放在后面就会出问题  精度问题

            similarity_fg = (similarity_fg * 255).astype(np.uint8)
            similarity_bg = (similarity_bg * 255).astype(np.uint8)

            
            similarity_fg = cv2.applyColorMap(similarity_fg, cv2.COLORMAP_JET)
            similarity_bg = cv2.applyColorMap(similarity_bg, cv2.COLORMAP_JET)
            combine_sim = cv2.applyColorMap(combine_sim, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(base_dir, "similarity_fg.png"),similarity_fg)
            cv2.imwrite(os.path.join(base_dir, "similarity_bg.png"),similarity_bg)
            cv2.imwrite(os.path.join(base_dir, "combine_sim_no10time.png"),combine_sim)
            cv2.imwrite(os.path.join(base_dir, "combine_sim_after.png"),combine_sim_after)
            

        pred[pred == 1] = cls
        pred = pred.permute(1, 2, 0)
        pred = pred.cpu().numpy()


        pred_view = pred.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(base_dir, "pred_view.png"), pred_view)
        # mask_q[mask_q == 1] = cls
        # print(np.unique(mask_q))
        # print(np.unique(pred))
        mask_t = pred.reshape((orininal_h, -1)) > 0
        seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pred, [-1])], [orininal_h, orininal_w, -1])
        image = Image.fromarray(np.uint8(seg_img))


        origin_img_path = os.path.join(args.data_root, 'JPEGImages')
        origin_image = Image.open(os.path.join(origin_img_path, id_q[0] + ".jpg")).convert('RGB')
        origin_image = np.array(origin_image)
        # imd_image = origin_image * mask_t
        # image = Image.blend(origin_image, image, 0.7)
        # image = seg_img + imd_image
        origin_image[mask_t] = seg_img[mask_t]
        # origin_image = Image.fromarray(origin_image)
        # image = Image.blend(origin_image, image, 0.7)

        image = Image.fromarray(origin_image)
        
        image.save(os.path.join(base_dir, id_q[0]+".jpg"))



def main():
    args = parse_args()
    testset = FewShot(args.dataset, args.data_root, None, 'val',
                      args.fold, args.shot, 1000)
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=4, drop_last=False)
    model = MatchingNet(args.backbone)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    model.cuda()
    evaluate(model, testloader, args=args)


def feature_vis(output_shape, savedir, level, feats): # feaats形状: [b,c,h,w]
    #  output_shape = (460, 460) # 输出形状
     channel_mean = torch.mean(feats,dim=1,keepdim=True) # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
     channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
     channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().numpy() # 四维压缩为二维
     channel_mean = (((channel_mean - np.min(channel_mean))/(np.max(channel_mean)-np.min(channel_mean)))*255).astype(np.uint8)
     if not os.path.exists(savedir+'/feature_vis'): os.makedirs(savedir+'/feature_vis')
     channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
     cv2.imwrite(savedir+'/feature_vis/'+ level + '.png',channel_mean)

if __name__ == "__main__":
    main()
    # feature = torch.randn((1, 512, 38, 38))
    # feature_vis(feature)