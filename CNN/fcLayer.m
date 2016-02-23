classdef fcLayer

    properties
        name;
        num_input;
        num_output;

        learning_rate;
        weight_decay;
        momentum;

        input;
        input_shape;
        output;
        W;
        b;
       
        grad_W;
        grad_b;
        diff_W; % last update for W
        diff_b; % last update for b
        delta;
    end

    methods
        function layer = fcLayer(name, num_input, num_output, init_std, learning_rate, weight_decay, momentum)
            layer.name = name;
            layer.num_input = num_input;
            layer.num_output = num_output;

            layer.W = single(random('norm', 0, init_std, num_output, num_input));
            layer.b = zeros(num_output, 1, 'single');
            layer.diff_W = zeros(size(layer.W), 'single');
            layer.diff_b = zeros(size(layer.b), 'single');

            layer.learning_rate = learning_rate;
            layer.weight_decay = weight_decay;
            layer.momentum = momentum;
        end

        function layer = forward(layer, input)
            % Your codes here
            [h,w,d,batch] = size(input);
            layer.input_shape = size(input);
            % input is 4-d, conform it into 2-d
            layer.input = reshape(input,[h*w*d,batch]);
            B = repmat(layer.b,1,batch);
            % output: n_out x batch
            layer.output = layer.W * layer.input + B;
        end

        function layer = backprop(layer, delta)
            % Your codes here
            % delta: n_out x batch
            % layer.delta: n_in x batch = (h*w*d) x batch
            layer.delta = layer.W'*delta;
            % convert 2-d delta to 4-d delta for previous layers
            layer.delta = reshape(layer.delta,layer.input_shape);
            layer.grad_W = delta * layer.input';
            layer.grad_b = sum(delta,2);

            layer.diff_W = layer.momentum * layer.diff_W - layer.learning_rate * (layer.grad_W + layer.weight_decay * layer.W);
            layer.W = layer.W + layer.diff_W;

            layer.diff_b = layer.momentum * layer.diff_b - layer.learning_rate * (layer.grad_b + layer.weight_decay * layer.b);
            layer.b = layer.b + layer.diff_b;
        end
    end
end