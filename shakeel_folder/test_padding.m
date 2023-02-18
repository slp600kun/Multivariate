fs = 16000;
samples = [1,10*fs];
maxlen = 10*fs;
thisFile = "17_52.wav";
[x,fs] = audioread(thisFile);
y =x;
x = x/sqrt(sum(abs(x.^2))/length(x));
if (length(x) < 10*fs)
    x(end+1:maxlen) = 0;
else
    x = x(1:maxlen);
end
x = [x;x(1:10*fs-length(x))];
feat = mel_spectrogram_bad(x, fs);
D = strsplit(thisFile,'.');
save(D{1},'feat');