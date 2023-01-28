import logging
import telebot
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from PIL import Image

from config import NUM_STEPS
from modules.loss import Normalization, ContentLoss, StyleLoss


class StyleModel(object):
    MAX_IMG_SIZE = (650, 650)
    CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406])
    CNN_NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225])
    RESIZE_IMAGES = False
    MAX_WORKERS = 1

    def __init__(self, bot: telebot.TeleBot) -> None:
        self.bot: telebot.TeleBot = bot
        self.active_tasks: int = 0
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.cnn_normalization_mean = self.CNN_NORMALIZATION_MEAN.to(self.device)
        self.cnn_normalization_std = self.CNN_NORMALIZATION_STD.to(self.device)
        self.unloader = transforms.ToPILImage()

        logging.info("Model created")

    def image_loader(self, image_name: str, imgsize: list[int, int]) -> torch.Tensor:
        if self.RESIZE_IMAGES:
            if imgsize[0] > self.MAX_IMG_SIZE[0]:
                imgsize[0] = int(imgsize[0] / (imgsize[0] / self.MAX_IMG_SIZE[0]))
                if int(imgsize[1] / (imgsize[0] / self.MAX_IMG_SIZE[0])) > 0:
                    imgsize[1] = int(imgsize[1] / (imgsize[0] / self.MAX_IMG_SIZE[0]))
            if imgsize[1] > self.MAX_IMG_SIZE[1]:
                imgsize[1] = int(imgsize[1] / (imgsize[1] / self.MAX_IMG_SIZE[1]))
                if int(imgsize[1] / (imgsize[1] / self.MAX_IMG_SIZE[1])) > 0:
                    imgsize[0] = int(imgsize[1] / (imgsize[1] / self.MAX_IMG_SIZE[1]))
        loader = transforms.Compose(
            [
                transforms.Resize(imgsize),
                transforms.ToTensor()
            ]
        )
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)
        logging.info(f"Uploaded file {image_name}, size: {tuple(imgsize)}")
        return image.to(self.device, torch.float)

    def imshow(self, tensor: torch.Tensor) -> Image.Image:
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.unloader(image)
        return image

    async def get_style_model_and_losses(
            self, normalization_mean, normalization_std,
            style_img, content_img,
            content_layers=None,
            style_layers=None
    ) -> tuple[nn.Sequential, list, list]:
        if style_layers is None:
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        if content_layers is None:
            content_layers = ['conv_6']
        normalization = Normalization(normalization_mean, normalization_std).to(self.device)

        content_losses = []
        style_losses = []

        sequential_model = nn.Sequential(normalization)

        indexes = 0
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d):
                indexes += 1
                name = 'conv_{}'.format(indexes)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(indexes)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(indexes)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(indexes)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            sequential_model.add_module(name, layer)

            if name in content_layers:
                target = sequential_model(content_img).detach()
                content_loss = ContentLoss(target)
                sequential_model.add_module("content_loss_{}".format(indexes), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = sequential_model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                sequential_model.add_module("style_loss_{}".format(indexes), style_loss)
                style_losses.append(style_loss)

        for indexes in range(len(sequential_model) - 1, -1, -1):
            if isinstance(sequential_model[indexes], ContentLoss) or \
                    isinstance(sequential_model[indexes], StyleLoss):
                break

        sequential_model = sequential_model[:(indexes + 1)]

        return sequential_model, style_losses, content_losses

    @staticmethod
    def get_input_optimizer(input_img):
        return optim.LBFGS([input_img])

    async def run_style_transfer(
            self, content_img, style_img, input_img, num_steps=300,
            style_weight=1000000, content_weight=1, chat_id: int = 0, message_id: int = 0
    ):
        self.active_tasks += 1
        logging.info('Building the style transfer model..')
        style_model, style_losses, content_losses = await self.get_style_model_and_losses(
            self.cnn_normalization_mean, self.cnn_normalization_std, style_img, content_img
        )

        input_img.requires_grad_(True)
        style_model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)

        logging.info('Optimizing..')
        self.bot.edit_message_text("Обработка..\nОптимизация (это может занять некоторое время)", chat_id, message_id)
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
                if run[0] % 10 == 0:
                    logging.info('Run : {} Style Loss : {:4f} Content Loss : {:4f}'.format(
                        run[0], style_score.item(), content_score.item()
                        )
                    )
                    self.bot.edit_message_text(f"Обработка..\n{run[0]}/{NUM_STEPS}", chat_id, message_id)
                if run[0] % 50 == 0 and run[0] != 0 and run[0] != num_steps:
                    with torch.no_grad():
                        prom_input = torch.clone(input_img)
                        prom_input.clamp_(0, 1)
                        self.imshow(prom_input).save(f"user_files/result_{chat_id}.jpg")
                        self.bot.send_photo(
                            chat_id, photo=open(f'user_files/result_{chat_id}.jpg', 'rb'), caption=f"{run[0]}/{NUM_STEPS}"
                        )
                return style_score + content_score

            optimizer.step(closure)
        self.bot.edit_message_text("Обработка окончена", chat_id, message_id)
        with torch.no_grad():
            input_img.clamp_(0, 1)
        self.active_tasks -= 1
        return input_img

