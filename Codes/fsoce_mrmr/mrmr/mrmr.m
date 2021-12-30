addpath(genpath('./'));
f_matrix = load('x_train_tmp.txt');
label = load('y_train_tmp.txt');
fea = mrmr_cal(f_matrix,label);
fea = fea -1; %this is for python index
save('mrmr_fse.txt','fea','-ascii');
