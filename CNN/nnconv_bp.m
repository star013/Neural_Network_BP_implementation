function [down_delta, grad_W, grad_b] = nnconv_bp(input, delta, W, kernel_size, pad)
    % Your codes here
    [delta_h,delta_w,n_out,~] = size(delta);
    % D_valid n_out x H2*W2*batch_size
    D_valid = delta_tensor2mtx(delta);
    grad_b = sum(D_valid,2);
    
    if pad>0
       input = padtensor(input, pad); 
    end
    % Y = H2*W2*batch_size x (H1-H2+1)*(W1-W2+1)*n_in
    %     = H2*W2*batch_size x K1*K2*n_in
    Y = y_tensor2mtx(input,delta_h,delta_w);
    grad_W = D_valid*Y;
    
    % reverse weight matrix W
    [Height,Width,n_in,B] = size(input);
    % W: n_out x k*k*n_in
    W_reverse = zeros(size(W));
    for k=1:n_in
       W_reverse(:,(k-1)*kernel_size*kernel_size+1:k*kernel_size*kernel_size) = fliplr(W(:,(k-1)*kernel_size*kernel_size+1:k*kernel_size*kernel_size)); 
    end
    W_reverse = mat2cell(W_reverse,ones(1,n_out),kernel_size*kernel_size*ones(1,n_in));
    W_reverse = cell2mat(W_reverse');
    % now W_reverse: n_in x k*k*n_out
    % delta_full: h1 x w1 x n_out x batch
    delta_full = padtensor(delta,kernel_size-1);
    % D_full: k*k*n_out x h1*w1*batch
    D_full = tensor2mtx(delta_full,kernel_size,kernel_size);
    down_delta = W_reverse*D_full;
    % down_delta: n_in x h1*w1*batch
    down_delta = mtx2tensor(down_delta,Height,Width,n_in,B);
    % shrink the down_delta as Height x Width
    % down_delta has the same size as input!
    down_delta = depadtensor(down_delta,pad);
end