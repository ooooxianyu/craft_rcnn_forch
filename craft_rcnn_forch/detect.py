from craft import CRAFT

import os
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import imgproc
import craft_utils
import numpy as np

from PIL import Image

import src.utils as utils
import src.dataset as dataset

import crnn.seq2seq as crnn

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

# arg detect
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--img_path', default='data/test_img/test6.jpg', type=str, help='input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')

# arg rec
parser.add_argument('--img_height', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--img_width', type=int, default=280, help='the width of the input image to network')
parser.add_argument('--hidden_size', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--encoder', type=str, default='weights/encoder_1.pth', help="path to encoder (to continue training)")
parser.add_argument('--decoder', type=str, default='weights/decoder_1.pth', help='path to decoder (to continue training)')
parser.add_argument('--max_width', type=int, default=71, help='the width of the feature map out from cnn')
parser.add_argument('--use_gpu', action='store_true', help='whether use gpu')

args = parser.parse_args()

def seq2seq_decode(encoder_out, decoder, decoder_input, decoder_hidden, max_length):
    decoded_words = []
    prob = 1.0
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_out)
        probs = torch.exp(decoder_output)
        _, topi = decoder_output.data.topk(1)
        ni = topi.squeeze(1)
        decoder_input = ni
        prob *= probs[:, ni]
        if ni == utils.EOS_TOKEN:
            break
        else:
            decoded_words.append(converter.decode(ni))

    words = ''.join(decoded_words)
    prob = prob.item()

    return words, prob

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

if __name__ == '__main__':

    # load alphabet
    with open('./data/char_std_5990.txt', encoding="UTF-8") as f:
        data = f.readlines()
        alphabet = [x.rstrip() for x in data]
        alphabet = ''.join(alphabet)

    # define convert bwteen string and label index
    converter = utils.ConvertBetweenStringAndLabel(alphabet)

    # len(alphabet) + SOS_TOKEN + EOS_TOKEN
    num_classes = len(alphabet) + 2

    transformer = dataset.ResizeNormalize(img_width=args.img_width, img_height=args.img_height)

    # load detect net
    detect_net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        detect_net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        detect_net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        detect_net = detect_net.cuda()
        detect_net = torch.nn.DataParallel(detect_net)
        cudnn.benchmark = False

    detect_net.eval()

    # load rec_net
    encoder = crnn.Encoder(3, args.hidden_size)
    # no dropout during inference
    decoder = crnn.Decoder(args.hidden_size, num_classes, dropout_p=0.0, max_length=args.max_width)
    print(encoder)
    print(decoder)
    if torch.cuda.is_available() and args.use_gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        map_location = 'cuda'
    else:
        map_location = 'cpu'

    encoder.load_state_dict(torch.load(args.encoder, map_location=map_location))
    print('loading pretrained encoder models from {}.'.format(args.encoder))
    decoder.load_state_dict(torch.load(args.decoder, map_location=map_location))
    print('loading pretrained decoder models from {}.'.format(args.decoder))

    encoder.eval()
    decoder.eval()


    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    #img_path = 'data/test_img/test1.jpg'
    #image = imgproc.loadImage(img_path)
    image = cv2.imread(args.img_path,1)
    #print(image.shape)
    bboxes, polys, score_text = test_net(detect_net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda,args.poly, refine_net)
    #print(bboxes.shape)
    for box in bboxes:

        detect_img = image[int(box[0,1]):int(box[2,1]),int(box[0,0]):int(box[2,0])]
        detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
        _detect_img = transformer(detect_img)

        _detect_img = _detect_img.view(1, *_detect_img.size())
        _detect_img = torch.autograd.Variable(_detect_img)

        encoder_out = encoder(_detect_img)

        max_length = 20
        decoder_input = torch.zeros(1).long()
        decoder_hidden = decoder.initHidden(1)
        if torch.cuda.is_available() and args.use_gpu:
            decoder_input = decoder_input.cuda()
            decoder_hidden = decoder_hidden.cuda()

        words, prob = seq2seq_decode(encoder_out, decoder, decoder_input, decoder_hidden, max_length)
        print('predict_string: {} => predict_probility: {}'.format(words, prob))

        cv2.imshow("1",detect_img)
        cv2.waitKey(0)

