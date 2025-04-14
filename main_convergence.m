close all;
clear;
clc;

addpath('data');
addpath('utility');

%---------------------- load data------------------------------------------
missing_raitos = [0, 0.1, 0.3, 0.5];
ratio_len = length(missing_raitos);

max_iter1 = 200;
dataset_len = 6;
obj1_iters = zeros(dataset_len, ratio_len);
obj1_values = zeros(dataset_len, ratio_len, max_iter1);

index_set = [1, 2, 3, 4, 5];
% index_set = [1, 2];
for s_index = 1 : length(index_set)
    data_index = index_set(s_index);
    switch data_index
        case 1
            filename = "MSRCv1";
            load('MSRCv1.mat');
            n = length(Y);
            nv = size(X, 2);
            K = length(unique(Y));
            gnd = Y;
            data_views = cell(1, nv);
            for nv_idx = 1 : nv      
                data_views{nv_idx} = X{nv_idx}';           
            end
            data_views = normalize_multiview_data(data_views);

            num_neighbors_set = [5, 5, 5, 15];
            alphas = [0.05, 0.05, 0.05, 0.01];
            betas = [0.05, 0.05, 0.01, 0.05];
    
        case 2
            filename = "BBC";
            load('BBC4view_685.mat');
            n = size(truelabel{1}, 2);
            nv = size(data, 2);
            K = length(unique(truelabel{1}));
            gnd = truelabel{1}';
            data_views = cell(1, nv);
            for nv_idx = 1 : nv
                 data_views{nv_idx} = data{nv_idx};
            end
            data_views = normalize_multiview_data(data_views);
            num_neighbors_set = [5, 10, 10, 5];
            alphas = [0.05, 0.05, 0.05, 0.05];
            betas = [0.01, 0.05, 0.05, 0.01];
      
        case 3
            filename = "flower17";
            load('flower17_Kmatrix.mat');
            n = length(Y);
            nv = size(KH, 3);
            K = length(unique(Y));
            gnd = Y;
            data_views = cell(1, nv);
            for nv_idx = 1 : nv
                 data_views{nv_idx} = KH(:, :, nv_idx);
            end
            data_views = normalize_multiview_data(data_views);  
            num_neighbors_set = [10, 10, 15, 20];
            alphas = [0.01, 0.01, 0.01, 0.005];
            betas = [0.001, 0.005, 0.01, 0.001];
    
        case 4
            filename = "handwritten";
            load('handwritten.mat');
            n = length(Y);
            nv = size(X, 2);
            K = length(unique(Y));
            gnd = Y + 1;
            data_views = cell(1, nv);
            for nv_idx = 1 : nv
                 data_views{nv_idx} = X{nv_idx}';
            end
            num_neighbors_set = [5, 5, 5, 5];
            alphas = [0.05, 0.05, 0.05, 0.05];
            betas = [0.05, 0.05, 0.05, 0.1];
    
          case 5
            filename = "NUS";
            load('NUS.mat');
            n = length(Y);
            nv = size(X, 2);
            K = length(unique(Y));
            gnd = Y;
            data_views = cell(1, nv);
            for nv_idx = 1 : nv
                 data_views{nv_idx} = X{nv_idx}';
            end
            data_views = normalize_multiview_data(data_views);
            num_neighbors_set = [20, 15, 10, 20];
            alphas = [0.005, 0.005, 0.01, 0.005];
            betas = [0.01, 0.01, 0.005, 0.01];
    
        case 6
            filename = "Caltech101";
            load('Caltech101.mat');
            nv = size(fea, 2);
            gnd = gt;
    
            %remove the background category
            positions = find(gnd > 1);
            gnd = gnd(positions);
            K = length(unique(gnd));
            gnd = gnd - 1;
            n = length(gnd);
    
            data_views = cell(1, nv);
            for nv_idx = 1 : nv
                tmp = fea{nv_idx}';
                data_views{nv_idx} = tmp(:, positions);            
            end
            data_views = normalize_multiview_data(data_views);  
            num_neighbors_set = [5, 10, 5, 10];
            alphas = [0.005, 0.005, 0.005, 0.005];
            betas = [0.001, 0.001, 0.001, 0.001];
    end       

    final_result = strcat(filename, '_result.txt');
    class_labels = zeros(1, K);
    for idx =  1 : K
        class_labels(idx) = length(find(gnd == idx));
    end  
            
    Mn = cell(1, nv);
    for raito_idx = 1 : length(missing_raitos)
        % prepare for incomplete multiview data: a set of the incomplete data instances 
        stream = RandStream.getGlobalStream;
        reset(stream);
        missing_raito = missing_raitos(raito_idx);
        raito = 1 - missing_raito;    
        rand('state', 1);
        for nv_idx = 1 : nv        
            if raito < 1
                pos = randperm(n);
                num = floor(n * raito);
                sample_pos = zeros(1, n);
                % 1 represents the corresponding feature available
                sample_pos(pos(1 : num)) = 1; 
                Mn{nv_idx} = sample_pos;
            else
                Mn{nv_idx} = ones(1, n);
            end
        end
    
        num_neighbors = num_neighbors_set(raito_idx);
        alpha = alphas(raito_idx); 
        beta = betas(raito_idx);
        tic;
        [Z_views, H_views, Y] = data_preprocess(data_views, Mn, num_neighbors, K);
        [Y, iter1, fobj1] = oagl(Z_views, H_views, Y, alpha, beta);
        time_cost = toc;

        obj1_iters(data_index, raito_idx) = iter1;
        obj1_values(data_index, raito_idx, :) = fobj1;

        try
            [~ , labels] = max(Y, [], 2);
            acc = accuracy(gnd, labels);
            cluster_data = cell(1, K);
            for pos_idx =  1 : K
                cluster_data(1, pos_idx) = { gnd(labels == pos_idx)' };
            end
            [nmi, purity, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data);
            disp([missing_raito, num_neighbors, alpha, beta, acc, nmi, purity, fmeasure, ri, ari, iter1]);            
            writematrix([missing_raito, num_neighbors, alpha, beta, roundn(acc, -2), roundn(nmi, -4), roundn(purity, -4), roundn(fmeasure, -4), roundn(ri, -4), roundn(ari, -4), roundn(time_cost, -2), iter1], final_result, "Delimiter", 'tab', 'WriteMode', 'append');       
        catch
            writematrix([missing_raito, num_neighbors, alpha, beta, iter1], final_result, "Delimiter", 'tab', 'WriteMode', 'append');
        end
    end
end

save('final_conv_results.mat', 'missing_raitos', 'obj1_iters', 'obj1_values');

