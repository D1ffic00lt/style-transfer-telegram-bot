import os
import cv2
import warnings
import telebot
import logging
import asyncio
import time

from config import (
    TOKEN, FORMAT,
    DATE_FORMAT, LOG_PATH, NUM_STEPS
)
from modules.model import StyleModel

warnings.filterwarnings("ignore")

bot = telebot.TeleBot(TOKEN)
bot.skip_pending = True

logging.basicConfig(format=FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)
handler = logging.FileHandler(LOG_PATH, mode='+a')
handler.setFormatter(logging.Formatter(FORMAT))
logging.getLogger().addHandler(handler)

model = StyleModel(bot)

logging.info("Program started")


@bot.message_handler(commands=["start", "convert"])
def start(message: telebot.types.Message) -> None:
    if model.active_tasks >= model.MAX_WORKERS:
        bot.send_message(message.chat.id, "Обработка уже выполняется!")
    else:
        bot.send_message(message.chat.id, "Отправьте стиль!")
        bot.register_next_step_handler(message, get_style)


def get_style(message: telebot.types.Message) -> None:
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


def get_object(message: telebot.types.Message) -> None:
    if message.photo is None:
        return
    if not os.path.isdir("user_files"):
        os.mkdir("user_files")

    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    with open('user_files/' + "object_" + str(message.chat.id) + ".jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    message_from_bot = bot.send_message(message.chat.id, "Обработка")

    image_shape = list(cv2.imread(f"user_files/object_{message.chat.id}.jpg").shape[:2])
    style_img = model.image_loader(f"user_files/style_{message.chat.id}.jpg", image_shape)
    content_img = model.image_loader(f"user_files/object_{message.chat.id}.jpg", image_shape)
    input_img = content_img.clone()

    output = asyncio.run(
        model.run_style_transfer(
            content_img, style_img,
            input_img, num_steps=NUM_STEPS,
            message_id=message_from_bot.id, chat_id=message.chat.id
        )
    )
    output = model.imshow(output)
    output.save(f"user_files/result_{message.chat.id}.jpg")

    bot.send_photo(message.chat.id, photo=open(f'user_files/result_{message.chat.id}.jpg', 'rb'))

    logging.info("Tensor converted to image")

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
        time.sleep(5)
