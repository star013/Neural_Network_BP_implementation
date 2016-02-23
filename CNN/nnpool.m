function output = nnpool(input, kernel_size, pad)
    % Your codes here
    if pad>0
       input = padtensor(input, pad); 
    end
    [h,w,n_in,batch] = size(input);
    % add some zeros at the right and bottom side in order to keep the
    % number of pool integer
    h_add = mod(h,kernel_size);
    w_add = mod(w,kernel_size);
    temp = zeros(h+h_add,w+w_add,n_in,batch);
    temp(1:h,1:w,:,:) = input;
    temp = reshape(temp,[h+h_add,w+w_add,n_in*batch]);
    mtx_height = kernel_size*kernel_size;
    mtx_width = (h+h_add)/kernel_size*(w+w_add)/kernel_size;
    mtx = zeros(mtx_height,mtx_width*batch*n_in);
    for n = 1:batch*n_in
      mtx(:,(n-1)*mtx_width+1:n*mtx_width)=im2col(temp(:,:,n),[kernel_size,kernel_size],'distinct'); 
    end
    mtx_ave = mean(mtx);
    output = reshape(mtx_ave,[(h_add+h)/kernel_size,(w+w_add)/kernel_size,n_in,batch]);
end