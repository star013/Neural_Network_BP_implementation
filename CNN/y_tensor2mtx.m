function [ mtx ] = y_tensor2mtx( tensor, H2, W2 )
% Convert a 4-dimensional tensor to a 2-dimensional matrix
% This is the prerequisite step of matrix multiplication instead of
% convolutional operation
% mtx = H2*W2*batch_size x (H1-H2+1)*(W1-W2+1)*n_in
%     = H2*W2*batch_size x K1*K2*n_in
[H1,W1,n_in, batch_size] = size(tensor);
mtx_height = H2*W2;
mtx_width = (H1-H2+1)*(W1-W2+1);
mtx = zeros(mtx_height*batch_size,mtx_width*n_in);
for m = 1:n_in
   for n = 1:batch_size
      mtx((n-1)*mtx_height+1:n*mtx_height,(m-1)*mtx_width+1:m*mtx_width)=im2col(tensor(:,:,m,n),[H2,W2],'sliding'); 
   end
end


end


