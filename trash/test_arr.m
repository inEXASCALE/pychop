addpath("chop")
data = load('verify.mat');
data = data.X;  % Access the array using the dictionary-style notation


options.format = 'h';
options.round = 1;
options.subnormal = 0;
chop([],options)


tic;
emu_val = chop(data);
toc;

disp(emu_val(1:10, 1:5));