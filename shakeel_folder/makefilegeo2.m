cd 'E:\research-data\multimodal\uptill-now\forged-geophone\2019-11-02-04-00-00\'
datFiles = dir("*.dat"); 
N = length(datFiles) ;
% loop for each file
t = 10;
k = 1;
for j = 1:270
    f = 1;
    for i = k:t
        thisFile = datFiles(i).name ;
        z(:,f) = importfile(thisFile, 2, 4001);
        f = f+1;
    end
    A = table2array(z);
    zvector = reshape(A, [], 1);
    addpath('E:\research-data\multimodal\datamaking\2019-11-20-18-10-40\Dependency\voicebox\')
    fsg=4000; % sampling frequency in Hz
    tsg=1/fsg; % sampling interval in seconds
    velocity = 0.0047*10^(-9);
    zaxisvelocity = velocity.*zvector;
    velocitysignal = detrend(zaxisvelocity);
    zaxisvelocitymeanremove = detrend(zaxisvelocity);

    feat = melcepst(zaxisvelocitymeanremove,fsg,'t',22); % 22 filters in mel-domain

    C = strsplit(thisFile,'.');

    save(C{1},'feat');
    k = k+10;
    t = t+10;
    clear z A zvector zaxisvelocity velocitysignal zaxisvelocitymeanremove feat C f
end