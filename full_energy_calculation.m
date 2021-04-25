filename = 'output_string.wav';
[y,Fs] = audioread(filename);
yc = movstd(y,1000);
plot(y,'b');
hold on;
plot(yc,'r');
title('输出音频及能量');
xlabel('长度');
ylabel('幅值');
legend('预测信号','能量信号')