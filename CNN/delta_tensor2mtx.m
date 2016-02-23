function [ mtx ] = delta_tensor2mtx(delta)
% Convert a 4-dimensional tensor to a 2-dimensional matrix
% This is the prerequisite step of matrix multiplication instead of
% convolutional operation
% mtx = n_out x H2*W2*batch_size
[H,W,n_out, batch_size] = size(delta);
delta = reshape(delta,[H*W,n_out,batch_size]);
mtx = zeros(n_out,H*W*batch_size);
mtx_width = H*W;
for m = 1:n_out
   for n = 1:batch_size
      mtx(m,(n-1)*mtx_width+1:n*mtx_width)=delta(:,m,n);
   end
end


end

