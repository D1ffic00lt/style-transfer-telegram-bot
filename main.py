import os
import warnings
import telebot
import logging
import asyncio

from config import (
    TOKEN, FORMAT,
    DATE_FORMAT, LOG_PATH
)
from modules.model.model import StyleModel

warnings.filterwarnings("ignore")

bot = telebot.TeleBot(TOKEN)
bot.skip_pending = True


logging.basicConfig(format=FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)
handler = logging.FileHandler(LOG_PATH, mode='+a')
handler.setFormatter(logging.Formatter(FORMAT))
logging.getLogger().addHandler(handler)


model = StyleModel()

logging.info("Program started")


@bot.message_handler(commands=["start", "convert"])
def start(message: telebot.types.Message):
    bot.send_message(message.chat.id, "Отправьте стиль!")
    bot.register_next_step_handler(message, get_style)


def get_style(message: telebot.types.Message):
    if message.photo is None:
        return
    if not os.path.isdir("user_files"):
        os.mkdir("user_files")
    file_info = bot.get_file(message.photo[-1].file_id)
    logging.info(file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)
    src = 'user_files/' + "style_" + str(message.chat.id) + ".jpg"
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    bot.reply_to(message, "Отправьте объект")
    bot.register_next_step_handler(message, get_object)


def get_object(message: telebot.types.Message):
    logging.info(1)
    if message.photo is None:
        return
    if not os.path.isdir("user_files"):
        os.mkdir("user_files")
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    src = 'user_files/' + "object_" + str(message.chat.id) + ".jpg"
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)

    bot.send_message(message.chat.id, "Обработка")

    style_img = model.image_loader(f"user_files/style_{message.chat.id}.jpg")
    content_img = model.image_loader(f"user_files/object_{message.chat.id}.jpg")
    input_img = content_img.clone()
    logging.info("cloned")
    output = model.imshow(model.run_style_transfer(content_img, style_img, input_img, num_steps=400))
    logging.info("run_style_transfer")

    output.save(f"user_files/result_{message.chat.id}.jpg")

    bot.send_photo(message.chat.id, photo=open(f'user_files/result_{message.chat.id}.jpg', 'rb'))

    try:
        os.remove(f"user_files/style_{message.chat.id}.jpg")
        os.remove(f"user_files/object_{message.chat.id}.jpg")
        os.remove(f"user_files/result_{message.chat.id}.jpg")
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    try:
        bot.polling(non_stop=True)
    except Exception as ex:
        if not isinstance(ex, telebot.apihelper.ApiTelegramException):
            logging.error(ex, exc_info=True)
