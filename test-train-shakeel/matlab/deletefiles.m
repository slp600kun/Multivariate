N = length(E) ;
% deleting the file 
for i = 1:N
    thisFile = E(i).name ;
    delete(thisFile)
end