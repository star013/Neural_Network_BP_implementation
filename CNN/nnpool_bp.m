function down_delta = nnpool_bp(input, delta, kernel_size, pad)
    % Your codes here
    [h,w,n_in,batch] = size(input);
    [h_del,w_del,~,~] = size(delta);
    delta = reshape(delta,[h_del,w_del,n_in*batch]);
    % use 3-d form to reduce for loop
    recover = zeros(h_del*kernel_size,w_del*kernel_size,n_in*batch);
    temp = ones(kernel_size,kernel_size);
    for m = 1:n_in*batch
       recover(:,:,m) = kron(delta(:,:,m),temp); 
    end
    % recover to 4-d form
    recover = reshape(recover,[h_del*kernel_size,w_del*kernel_size,n_in,batch]);
    down_delta = recover(pad+1:h+pad,pad+1:w+pad,:,:)/kernel_size/kernel_size;
end