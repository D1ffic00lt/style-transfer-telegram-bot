import logging
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from PIL import Image

from modules.model.loss import Normalization, ContentLoss, StyleLoss


class StyleModel(object):
    IMGSIZE = (512, 512)
    CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406])
    CNN_NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self):
        self.sequential_model = None
        self.style_loss = None
        self.target_feature = None
        self.content_loss = None
        self.target = None
        self.name = None
        self.indexes = None
        self.style_losses = None
        self.content_losses = None
        self.normalization = None
        self.image = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.loader = transforms.Compose(
            [
                transforms.Resize(StyleModel.IMGSIZE),
                transforms.ToTensor()
            ]
        )
        self.cnn_normalization_mean = StyleModel.CNN_NORMALIZATION_MEAN.to(self.device)
        self.cnn_normalization_std = StyleModel.CNN_NORMALIZATION_STD.to(self.device)
        self.unloader = transforms.ToPILImage()
        logging.info("Model created")

    def image_loader(self, image_name):
        self.image = Image.open(image_name)
        self.image = self.loader(self.image).unsqueeze(0)
        logging.info(f"Uploaded file {image_name}")
        return self.image.to(self.device, torch.float)

    def imshow(self, tensor):
        self.image = tensor.cpu().clone()
        self.image = self.image.squeeze(0)
        self.image = self.unloader(self.image)
        logging.info(f"Tensor converted to image")
        return self.image

    async def get_style_model_and_losses(
            self, normalization_mean, normalization_std,
            style_img, content_img,
            content_layers=None,
            style_layers=None
    ):
        if style_layers is None:
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        if content_layers is None:
            content_layers = ['conv_4']
        self.normalization = Normalization(normalization_mean, normalization_std).to(self.device)

        self.content_losses = []
        self.style_losses = []

        self.sequential_model = nn.Sequential(self.normalization)

        self.indexes = 0
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d):
                self.indexes += 1
                self.name = 'conv_{}'.format(self.indexes)
            elif isinstance(layer, nn.ReLU):
                self.name = 'relu_{}'.format(self.indexes)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                self.name = 'pool_{}'.format(self.indexes)
            elif isinstance(layer, nn.BatchNorm2d):
                self.name = 'bn_{}'.format(self.indexes)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.sequential_model.add_module(self.name, layer)

            if self.name in content_layers:
                self.target = self.sequential_model(content_img).detach()
                self.content_loss = ContentLoss(self.target)
                self.sequential_model.add_module("content_loss_{}".format(self.indexes), self.content_loss)
                self.content_losses.append(self.content_loss)

            if self.name in style_layers:
                self.target_feature = self.sequential_model(style_img).detach()
                self.style_loss = StyleLoss(self.target_feature)
                self.sequential_model.add_module("style_loss_{}".format(self.indexes), self.style_loss)
                self.style_losses.append(self.style_loss)

        for self.indexes in range(len(self.sequential_model) - 1, -1, -1):
            if isinstance(self.sequential_model[self.indexes], ContentLoss) or \
                    isinstance(self.sequential_model[self.indexes], StyleLoss):
                break

        self.sequential_model = self.sequential_model[:(self.indexes + 1)]

        return self.sequential_model, self.style_losses, self.content_losses

    @staticmethod
    def get_input_optimizer(input_img):
        return optim.LBFGS([input_img])

    async def run_style_transfer(
            self, content_img, style_img, input_img, num_steps=300,
            style_weight=1000000, content_weight=1
    ):
        logging.info('Building the style transfer model..')
        style_model, style_losses, content_losses = await self.get_style_model_and_losses(
            self.cnn_normalization_mean, self.cnn_normalization_std, style_img, content_img
        )
        logging.info(str(type(style_losses)) + str(type(style_model)) + str(type(content_losses)))

        input_img.requires_grad_(True)
        style_model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)

        logging.info('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                style_model(input_img)
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
                    logging.info('Style Loss : {:4f} Content Loss : {:4f}'.format(
                        style_score.item(), content_score.item()
                        )
                    )

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img
