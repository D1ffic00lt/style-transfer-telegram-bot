<h1 align="center">Style Transfer Telegram Bot</h1>
<h2 align="center">Телеграм бот для переноса стиля изображения</h2>

<h2 align="center"><a href="https://www.mozilla.org/ru/](https://t.me/foto_creator_nn_bot)">Ссылка на бота</a></p>

Асинхронный телеграм бот на языке Python для переноса стиля изображения при помощи нейронной сети.

## Примеры
<p>
    <img src="https://media.discordapp.net/attachments/572705890524725248/1068940296802549942/Fotoram.io.jpg" width="370">
    <img src="https://media.discordapp.net/attachments/572705890524725248/1068940947104202782/Fotoram.io_1.jpg" width="370">    
</p>

## Файлы и директории

| Имя файла или директории             | Описание                                                                      |
|--------------------------------------|-------------------------------------------------------------------------------|
| [main.py](main.py)                   | Основной код                                                                  |
| [config.py](config.py)               | Файл конфигурации <br/>(формат даты, конфигурация модуля logging, токен бота) |
| [modules/loss.py](modules/loss.py)   | ContentLoss, StyleLoss и Normalization                                        |
| [modules/model.py](modules/model.py) | Код модели                                                                    |

## Использование

1. Отправьте команду **/start** или **/convert**
2. Отправьте фотографию стиля
<img src="https://user-images.githubusercontent.com/69642892/215318213-dd1b1aa4-e7e0-476c-91b6-cc307830e583.png" width="700">
3. Отправьте фотографию объекта
<img src="https://media.discordapp.net/attachments/572705890524725248/1069191182598557736/image.png" width="700">
4. Получите результат!
<img src="https://media.discordapp.net/attachments/572705890524725248/1069191281567342682/image.png" width="700">
