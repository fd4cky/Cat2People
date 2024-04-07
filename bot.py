import asyncio
from os import remove
from aiogram import Bot, Dispatcher, F
from aiogram.filters.state import State, StatesGroup
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from aiogram.fsm.context import FSMContext
from start import recognition, convert_ogg_to_wav

TOKEN = "TOKEN"
bot = Bot(TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher(bot=bot)

keyboardPanel = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True, keyboard=[
    [KeyboardButton(text="Распознать аудио")]
])

class Form(StatesGroup):
    audio = State()

@dp.message(CommandStart())
async def menu_output(message: Message):
    await message.answer("Добро пожаловать в бота для распознавания речи котов!", reply_markup=keyboardPanel)

@dp.message(F.text.lower() == "распознать аудио")
async def Audio(message: Message, state: FSMContext):
    await state.set_state(Form.audio)
    await message.answer("Запишите голосовое сообщение с мяуканьем вашего кота/кошки.", reply_markup=ReplyKeyboardRemove())

@dp.message(Form.audio)
async def GetAudio(message: Message, state: FSMContext):
    await state.update_data(audio=message.voice)
    path = f'voice/{message.voice.file_id}'
    try:
        msg = await message.answer(f"Обрабатывается..") 
        await bot.download(message.voice, destination=f"{path}.ogg")
        await convert_ogg_to_wav(path)
        await message.answer(recognition(f"{path}.wav"), reply_markup=keyboardPanel)
        await remove(f'{path}.ogg')
        await remove(f'{path}.wav')
        await msg.delete()
    except:
        await message.answer("Вы отправили не голосовое сообщение.", reply_markup=keyboardPanel)

async def main() -> None:
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())