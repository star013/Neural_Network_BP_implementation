classdef convLayer

    properties
        name;
        kernel_size;
        num_input;
        num_output;
        pad;

        learning_rate;
        weight_decay;
        momentum;

        input;
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
        function layer = convLayer(name, kernel_size, num_input, num_output, pad,...
                                   init_std, learning_rate, weight_decay, momentum)
            layer.name = name;
            layer.kernel_size = kernel_size;
            layer.num_input = num_input;
            layer.num_output = num_output;
            layer.pad = pad;

            layer.learning_rate = learning_rate;
            layer.weight_decay = weight_decay;
            layer.momentum = momentum;

            layer.W = single(random('norm', 0, init_std, num_output, kernel_size*kernel_size*num_input));
            layer.b = zeros(num_output, 1, 'single');
            layer.diff_W = zeros(size(layer.W), 'single');
            layer.diff_b = zeros(size(layer.b), 'single');
        end

        function layer = forward(layer, input)
            assert(size(input, 3) == layer.num_input, ['input channel is inconsistent with ' layer.name ' filter']);
            layer.input = input;
            layer.output = nnconv(layer.input, layer.kernel_size, layer.num_output,...
                layer.W, layer.b, layer.pad);
        end

        function layer = backprop(layer, delta)
            assert(isequal(size(delta), size(layer.output)), ['delta is inconsistent with ' layer.name ' output']);
            [layer.delta, layer.grad_W, layer.grad_b] = nnconv_bp(layer.input, delta, layer.W, layer.kernel_size, layer.pad);

            layer.diff_W = layer.momentum * layer.diff_W - layer.learning_rate * (layer.grad_W + layer.weight_decay * layer.W);
            layer.W = layer.W + layer.diff_W;

            layer.diff_b = layer.momentum * layer.diff_b - layer.learning_rate * (layer.grad_b + layer.weight_decay * layer.b);
            layer.b = layer.b + layer.diff_b;
        end
    end
end



