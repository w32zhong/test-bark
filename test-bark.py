# pip install git+https://github.com/suno-ai/bark.git && pip uninstall -y torch torchvision torchaudio && pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

text_prompt = """
刚才和你们聊的很开心， 王博士如果对 Bark 的效果感兴趣可以看一下这段 Bark 生成的语音：
全世界证券交易员，投资家的专业媒体
股市简报 内容来自全球专业媒体。 
由六度简报团队制作，六度简报的网址是: 6dobrief.com
对抗通胀的最后一英里可能是一场痛苦的跋涉 -- 悉尼先驱晨报
主要经济体的央行行长警告说，对抗通胀的斗争可能比最初预期的更加困难和持久。
在葡萄牙举行的欧洲央行论坛上，
包括美联储的 杰罗母 鲍威尔、欧洲央行的 christine lagarde 和英格兰银行的安德鲁贝利在内的央行主席都对通胀上升表示担忧。
"""
#text_prompt = """
#刚才和你们聊的很开心， 王博士如果对 Bark 的效果感兴趣可以看一下这段 Bark 生成的语音：
#"""

from bark import SAMPLE_RATE, generate_audio, preload_models
import numpy as np
from bark.api import semantic_to_waveform
import soundfile as sf

preload_models()

inputs = text_prompt.strip().split('\n')
silence = np.zeros(int(0.25 * SAMPLE_RATE))
GEN_TEMP = 0.6
SPEAKER = "v2/zh_speaker_6"

pieces = []
for input_line in inputs[:]:
  print('LINE:', input_line)
  audio_array = generate_audio(
        input_line,
        history_prompt=SPEAKER
    )
  pieces += [audio_array, silence.copy()]
cat = np.concatenate(pieces)

sf.write('output.wav', cat, SAMPLE_RATE, 'PCM_24')

#sudo pacman -S twolame sox
#sox *.wav output/output.wav
#ffmpeg -i output/output.wav -ab 320k -f mp3 output/output.mp3
