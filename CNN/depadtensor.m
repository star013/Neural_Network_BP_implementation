function [ output ] = depadtensor( tensor, pad )
% remove padding on tensor
[H,W,~,~] = size(tensor);
output = tensor(pad+1:H-pad,pad+1:W-pad,:,:);
end
