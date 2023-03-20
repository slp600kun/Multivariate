cd '\Users\slp60\Documents\MATLAB\multimodal\45_33\45_33\'
datFiles = dir("*.dat"); 
N = length(datFiles) ;
% loop for each file
t = 10;
k = 1;
f = 1;

for i = k:t
    thisFile = datFiles(i).name ;
    z(:,f) = importfile(thisFile, 2, 4001);
    f = f+1;
end
A = table2array(z);
zvector = reshape(A, [], 1);
addpath('\Users\slp60\Documents\MATLAB\multimodal\45_33\data_making')
fsg=4000; % sampling frequency in Hz
tsg=1/fsg; % sampling interval in seconds
velocity = 0.0047*10^(-9);
zaxisvelocity = velocity.*zvector;
%zaxisvelocitymeanremove = detrend(zaxisvelocity);

feat = melcepst(zaxisvelocity,fsg,'t',50,24,64,40); % 22 filters in mel-domain

C = strsplit(thisFile,'.');

save(C{1},'feat');
k = k+10;
t = t+10;
clear z A zvector zaxisvelocity velocitysignal zaxisvelocitymeanremove feat C f