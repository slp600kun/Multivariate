%A = subdir('*.mat');
N = length(A) ;
for i=1:N
  [pathname,filename,extension] = fileparts(A(i).name);
  % the new name, e.g.
  newPathname = 'E:\research-data\multimodal\new-features-999-50\true-geophone\';
  newFilename = ['1_g_',num2str(i),'_',filename];
  % rename the file
  copyfile(A(i).name, [newPathname newFilename extension]);
end