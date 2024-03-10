import discord
from discord.ext import commands
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np


def detect_guitar(image, model_path, labels_path):
    # отключаем научную нотацию
    np.set_printoptions(suppress=True)
    # загружаем модель
    model = load_model(model_path, compile=False)
    # загружаем лейблы (названия классов)
    class_names = open(labels_path, "r").readlines()
    # создаем массив нужной формы (нужна для подачи на вход модели в будущем)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # загрузка картинки
    image = Image.open(image).convert("RGB")
    # изменяем размеры изображения
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    # преобразование изображения к массиву
    image_array = np.asarray(image)
    # нормализация изображения (убираем шумы)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # в массив подставляем нормализированное изображение
    data[0] = normalized_image_array
    # получаем предсказание модели
    prediction = model.predict(data)
    # получаем индексч с наивысшим предсказанием класса
    index = np.argmax(prediction)
    # получаем имя класса
    class_name = class_names[index]
    # получаем вероятность этого класса
    confidence_score = prediction[0][index]
    # возвращем результат
    return ('Я думаю что это',class_name[2:],'c вероятностью',confidence_score)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

@bot.command()

async def check(ctx):
    # проверяем есть ли вложения
    if ctx.message.attachments:
        # перебираем каждое вложение
        for i in ctx.message.attachments:
            # берем имя файла
            file_name = i.filename
            # сохраняем файл
            await i.save(f'./{file_name}')
            await ctx.send(detect_guitar(f'./{file_name}', 'keras_model.h5', 'labels.txt'))
    else:
        await ctx.send('Вложений не было')


bot.run("MTE1MjkwNzMyMTAzNTQ2ODg3MQ.GjUUtB.161Q5ABXU-OgvhxM2p51trn5jzVpR3c_ARo0j0")