function [ tensor ] = mtx2tensor( mtx, h, w, n_out, batch_size )
% convert the result of Conv layer to proper former as next layer's input
% mtx = n_out x (H-K+1)(W-K+1)*batch_size
% h = H-K+1
% w = W-K+1
% tensor = H-K+1 x W-K+1 x n_out x batch_size
tensor = zeros(h,w,n_out,batch_size);
mtx = mtx';
for m=1:batch_size
    tensor(:,:,:,m)=reshape(mtx((m-1)*h*w+1:m*h*w,:),[h,w,n_out]); 
end

end

