<h1 align="center">Style Transfer Telegram Bot</h1>
<h2 align="center">Телеграм бот для переноса стиля изображения</h2>

Асинхронный телеграм бот на языке Python для переноса стиля изображения при помощи нейронной сети.

## Примеры

<img src="https://media.discordapp.net/attachments/572705890524725248/1068940296802549942/Fotoram.io.jpg" width="300">
<img src="https://media.discordapp.net/attachments/572705890524725248/1068940947104202782/Fotoram.io_1.jpg" width="300">

## Файлы и директории

| Имя файла или директории             | Описание                                                                      |
|--------------------------------------|-------------------------------------------------------------------------------|
| [main.py](main.py)                   | Основной код                                                                  |
| [config.py](config.py)               | Файл конфигурации <br/>(формат даты, конфигурация модуля logging, токен бота) |
| [modules/loss.py](modules/loss.py)   | ContentLoss, StyleLoss и Normalization                                        |
| [modules/model.py](modules/model.py) | Код модели                                                                    |

