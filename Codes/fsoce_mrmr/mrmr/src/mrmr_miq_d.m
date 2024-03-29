function [fea] = mrmr_miq_d(d, f, K)
% function [fea] = mrmr_miq_d(d, f, K)
% 
% The MIQ scheme of minimum redundancy maximal relevance (mRMR) feature selection
% 
% The parameters:
%  d - a N*M matrix, indicating N samples, each having M dimensions. Must be integers.
%  f - a N*1 matrix (vector), indicating the class/category of the N samples. Must be categorical.
%  K - the number of features need to be selected
%
% Note: This version only supports discretized data, thus if you have continuous data in "d", you 
%       will need to discretize them first. This function needs the mutual information computation 
%       toolbox written by the same author, downloadable at the Matlab source codes exchange site. 
%       Also There are multiple newer versions on the Hanchuan Peng's web site 
%       (http://research.janelia.org/peng/proj/mRMR/index.htm).
%
% More information can be found in the following papers. 
%
% H. Peng, F. Long, and C. Ding, 
%   "Feature selection based on mutual information: criteria 
%    of max-dependency, max-relevance, and min-redundancy,"
%   IEEE Transactions on Pattern Analysis and Machine Intelligence,
%   Vol. 27, No. 8, pp.1226-1238, 2005. 
%
% C. Ding, and H. Peng, 
%   "Minimum redundancy feature selection from microarray gene 
%    expression data,"  
%    Journal of Bioinformatics and Computational Biology,
%   Vol. 3, No. 2, pp.185-205, 2005. 
%
% C. Ding, and H. Peng, 
%   "Minimum redundancy feature selection from microarray gene 
%    expression data,"  
%   Proc. 2nd IEEE Computational Systems Bioinformatics Conference (CSB 2003),
%   pp.523-528, Stanford, CA, Aug, 2003.
%  
% By Hanchuan Peng (hanchuan.peng@gmail.com)
% April 16, 2003
%
d(isnan(d)) = 0;
nd = size(d,2);
%% step 1
t1=cputime;
for i=1:nd
   t(i) = mutualinfo(d(:,i), f);
end
fprintf('calculate the marginal dmi costs %5.1fs.\n', cputime-t1);
%%
[~, idxs] = sort(t, 'descend');
fea = [];
fea(1) = idxs(1);% first feature
KMAX = min(1000,nd); %500 %20000
idxleft = idxs(2:KMAX);
%% step 2
for k=2:K
   ncand = length(idxleft);
   curlastfea = length(fea);
   for i=1:ncand
      revelance_mi(i) = mutualinfo(d(:,idxleft(i)), f);% relevance
      mi_array(idxleft(i), curlastfea) = mutualinfo(d(:,fea(curlastfea)), d(:,idxleft(i)));
      redundancy_mi(i) = mean(mi_array(idxleft(i), :));% redundancy
   end
   [op, tmpidx] = max((revelance_mi(1:ncand) + eps) ./ (redundancy_mi(1:ncand) + eps));% Optimize miq
   % if op<1;return;end
   fea(k) = idxleft(tmpidx);idxleft(tmpidx) = [];
end
