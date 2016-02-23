function [ mtx ] = tensor2mtx( tensor, K1, K2 )
% Convert a 4-dimensional tensor to a 2-dimensional matrix
% This is the prerequisite step of matrix multiplication instead of
% convolutional operation
% mtx = K*K*n_in x (H-K+1)*(W-K+1)*batch_size
[H,W,n_in, batch_size] = size(tensor);
mtx_height = K1*K2;
mtx_width = (H-K1+1)*(W-K2+1);
mtx = zeros(mtx_height*n_in,mtx_width*batch_size);
for m = 1:n_in
   for n = 1:batch_size
      mtx((m-1)*mtx_height+1:m*mtx_height,(n-1)*mtx_width+1:n*mtx_width)=im2col(tensor(:,:,m,n),[K1,K2],'sliding'); 
   end
end


end

