function output = nnconv(input, kernel_size, num_output, W, b, pad)
    % Your codes here
    if pad>0
       input = padtensor(input, pad); 
    end
    [Height,Width,~,B] = size(input);
    height = Height-kernel_size+1;
    width = Width-kernel_size+1;
    A = tensor2mtx(input,kernel_size,kernel_size);
    output = W*A + repmat(b,1,height*width*B);
    output = mtx2tensor(output,height,width,num_output,B);
end