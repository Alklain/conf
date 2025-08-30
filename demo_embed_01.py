import torchaudio
import numpy as np
import pickle
import soundfile as sf
import matplotlib.pyplot as plt
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.speaker import SpeakerRecognition

path_1= 'bonch_16k.wav'
path_2= 'goblin_16k.wav'
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
[data, sr] = sf.read(path_1)


t1=4.5
t2=65.5
td = 5.5

[d1, sf1] = sf.read(path_1, start=int(sr*float(t1)), stop=int(sr*(float(t1)+float(td))))
sf.write('temp/temp.wav', d1, sf1)
signal, fs = torchaudio.load('temp/temp.wav')
embeddings1 = np.array(classifier.encode_batch(signal))




[d1, sf1] = sf.read(path_2, start=int(sr*float(t2)), stop=int(sr*(float(t2)+float(td))))
sf.write('temp/temp1.wav', d1, sf1)
signal1, fs = torchaudio.load('temp/temp1.wav')
embeddings2 = np.array(classifier.encode_batch(signal1))
print(embeddings1[0,0,:])


cosine_similarity = np.dot(embeddings1[0,0,:], embeddings2[0,0,:]) / (np.linalg.norm(embeddings1[0,0,:]) * np.linalg.norm(embeddings2[0,0,:]))
print(cosine_similarity)

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
score, prediction = verification.verify_files('temp/temp.wav', 'temp/temp1.wav')
print(prediction, score)


print('статистика ')
ts = np.linspace(5,60,100)
out = []
td = 5
for i in range(len(ts)):
    t1 = ts[i]
    t2 = ts[i]+65.5
    [d1, sf1] = sf.read(path_1, start=int(sr * float(t1)), stop=int(sr * (float(t1) + float(td))))
    sf.write('temp/temp.wav', d1, sf1)
    [d1, sf1] = sf.read(path_1, start=int(sr * float(t2)), stop=int(sr * (float(t2) + float(td))))
    sf.write('temp/temp1.wav', d1, sf1)
    score, prediction = verification.verify_files('temp/temp.wav', 'temp/temp1.wav')
    print(prediction, score)
    out.append(float(score))
out1 = []
for i in range(len(ts)):
    t1 = ts[i]
    t2 = ts[i]+65.5
    [d1, sf1] = sf.read(path_1, start=int(sr * float(t1)), stop=int(sr * (float(t1) + float(td))))
    sf.write('temp/temp.wav', d1, sf1)
    [d1, sf1] = sf.read(path_2, start=int(sr * float(t2)), stop=int(sr * (float(t2) + float(td))))
    sf.write('temp/temp1.wav', d1, sf1)
    score, prediction = verification.verify_files('temp/temp.wav', 'temp/temp1.wav')
    print(prediction, score)
    out1.append(float(score))
print(out1)
plt.figure()
plt.hist(out,15)
plt.hist(out1,15)
plt.show()