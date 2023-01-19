import logging

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from modules.model.loss import Normalization, ContentLoss, StyleLoss


class StyleModel(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.imsize = (256, 256)
        self.loader = transforms.Compose(
            [
                transforms.Resize(self.imsize),
                transforms.ToTensor()
            ]
        )
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.unloader = transforms.ToPILImage()
        logging.info("model inited")

    def image_loader(self, image_name):
        image = Image.open(image_name)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def imshow(self, tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.unloader(image)
        return image

    def get_style_model_and_losses(
            self, normalization_mean, normalization_std,
            style_img, content_img,
            content_layers=None,
            style_layers=None
    ):
        if style_layers is None:
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        if content_layers is None:
            content_layers = ['conv_4']
        normalization = Normalization(normalization_mean, normalization_std).to(self.device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    @staticmethod
    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer(
            self, content_img, style_img, input_img, num_steps=300,
            style_weight=1000000, content_weight=1
    ):
        logging.info('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(
            self.cnn_normalization_mean, self.cnn_normalization_std, style_img, content_img
        )

        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)

        logging.info('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    logging.info("run {}:".format(run))
                    logging.info('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img
