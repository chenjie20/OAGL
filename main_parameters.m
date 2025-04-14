close all;
clear;
clc;

addpath('data');
addpath('utility');

num_neighbors_set = [5, 10, 15, 20];
alphas = [1e-3, 5e-3, 0.01, 0.05];
betas = [1e-3, 5e-3, 0.01, 0.05];

%---------------------- load data------------------------------------------
% index_set = [1, 2, 3, 4, 5, 6];
index_set = [1];
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
    end
    
    final_result = strcat(filename, '_par_result.txt');
    final_mat_name = strcat(filename, '_par_labels.mat');
    class_labels = zeros(1, K);
    for idx =  1 : K
        class_labels(idx) = length(find(gnd == idx));
    end
    
    missing_raitos = [0, 0.1, 0.3, 0.5];
    neighbors_len = length(num_neighbors_set);
    alpha_length = length(alphas);
    beta_length = length(betas);
    ratio_len = length(missing_raitos);    
    
    final_accs = zeros(neighbors_len, alpha_length, beta_length, ratio_len);
    final_nmis = zeros(neighbors_len, alpha_length, beta_length, ratio_len);
    final_purities = zeros(neighbors_len, alpha_length, beta_length, ratio_len);
    final_fmeasures = zeros(neighbors_len, alpha_length, beta_length, ratio_len);
    final_ris = zeros(neighbors_len, alpha_length, beta_length, ratio_len);
    final_aris = zeros(neighbors_len, alpha_length, beta_length, ratio_len);
    final_costs = zeros(neighbors_len, alpha_length, beta_length, ratio_len);
    final_iters = zeros(neighbors_len, alpha_length, beta_length, ratio_len);
            
    Mn = cell(1, nv);
    for raito_idx = 2 : 3%length(missing_raitos)
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
    
        % the results on the different combinations of the parameters
        for num_neighbors_idx = 1 : length(num_neighbors_set)
            num_neighbors = num_neighbors_set(num_neighbors_idx);
            for alpha_idx = 1 : length(alphas)
                alpha = alphas(alpha_idx); 
                for beta_idx = 1 : length(betas)
                    beta = betas(beta_idx);                
                    tic;
                    [Z_views, H_views, Y] = data_preprocess(data_views, Mn, num_neighbors, K);
                    [Y, iter1, ~] = oagl(Z_views, H_views, Y, alpha, beta);
                    time_cost = toc;
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
    
                        final_accs(num_neighbors_idx, alpha_idx, beta_idx, raito_idx) = acc;
                        final_nmis(num_neighbors_idx, alpha_idx, beta_idx, raito_idx) = nmi;
                        final_purities(num_neighbors_idx, alpha_idx, beta_idx, raito_idx) = purity;                 
                        final_fmeasures(num_neighbors_idx, alpha_idx, beta_idx, raito_idx) = fmeasure;
                        final_ris(num_neighbors_idx, alpha_idx, beta_idx, raito_idx) = ri;
                        final_aris(num_neighbors_idx, alpha_idx, beta_idx, raito_idx) = ari;
                        final_costs(num_neighbors_idx, alpha_idx, beta_idx, raito_idx) = time_cost;
                        final_iters(num_neighbors_idx, alpha_idx, beta_idx, raito_idx) = iter1;
    
                    catch
                        disp('error');
                        writematrix([missing_raito, num_neighbors, alpha, beta, iter1], final_result, "Delimiter", 'tab', 'WriteMode', 'append');
                    end
                end
            end
        end
    end
    save(final_mat_name, 'num_neighbors_set', 'alphas', 'betas', 'final_accs', 'final_nmis', 'final_purities', 'final_fmeasures', 'final_ris', ...
            'final_aris', 'final_costs', 'final_iters');
end

