filename = 'output_string.wav';
[y,Fs] = audioread(filename);
yc = movstd(y,1000);
plot(y,'b');
hold on;
plot(yc,'r');
title('�����Ƶ������');
xlabel('����');
ylabel('��ֵ');
legend('Ԥ���ź�','�����ź�')