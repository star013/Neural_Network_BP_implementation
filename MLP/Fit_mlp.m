function net = Fit_mlp(net, train_data, train_label, test_data, test_label, ...
                       batchsize, epoch, display_freq, test_freq, snapshot_freq)
    now = clock;
    fprintf('[%02d:%02d:%05.2f] Start training\n', now(4), now(5), now(6));
    num_input = size(train_data, 2);
    num_test_input = size(test_data, 2);
    iters = ceil(num_input / batchsize);
    for k = 1:epoch
        loss = [];
        accuracy = [];
        for i = 1:iters
            net = Forward(net, train_data(:,(i-1)*batchsize+1:min(i*batchsize, num_input)));
            net = Backpropagation(net, train_label(:,(i-1)*batchsize+1:min(i*batchsize, num_input)));
            loss = [loss, net.layer_list{net.num_layer}.loss];
            accuracy = [accuracy, net.layer_list{net.num_layer}.accuracy];
        end 

        if mod(k, display_freq) == 0
            mean_loss = mean(loss);
            mean_accuracy = mean(accuracy);
            now = clock;
            fprintf('[%02d:%02d:%05.2f] epoch %d, training loss %.5f, accuracy %.4f\n',...
                now(4), now(5), now(6), k, mean_loss, mean_accuracy);
        end

        if mod(k, test_freq) == 0
            test_iters = ceil(num_test_input / batchsize);
            test_loss = [];
            test_accuracy = [];
            for j = 1:test_iters
                [lss, ~, acc] = Predict(net, test_data(:,(j-1)*batchsize+1:min(j*batchsize, num_test_input)),...
                    test_label(:,(j-1)*batchsize+1:min(j*batchsize, num_test_input)));
                test_loss = [test_loss, lss];
                test_accuracy = [test_accuracy, acc];
            end
            test_mean_loss = mean(test_loss);
            test_mean_accuracy = mean(test_accuracy);
            now = clock;
            fprintf('[%02d:%02d:%05.2f]   test results: loss %.5f, accuracy %.4f\n',...
                now(4), now(5), now(6), test_mean_loss, test_mean_accuracy);
        end

        if mod(k, snapshot_freq) == 0
            save(['net_iter_' num2str(k) '.mat'], 'net');
            now = clock;
            fprintf('[%02d:%02d:%05.2f] Snapshotting to net_iter_%d.mat\n',...
                now(4), now(5), now(6), k);
        end
    end
end
        