function [ output ] = padtensor( tensor, pad )
[H,W,D,B] = size(tensor);
output = zeros(H+2*pad,W+2*pad,D,B);
output(pad+1:H+pad,pad+1:W+pad,:,:) = tensor;
end

