function [Z_views, H_views, Y] = data_preprocess(data_views, Mn, kn, num_clusters)
% data_views{1}: each column of data_views{1} represents a sample
    num_sample = size(data_views{1}, 2);
    nv = size(data_views, 2);
    Z_views = cell(1, nv);
    H_views = cell(1, nv);
    Hs = zeros(num_sample, num_clusters);

    for nv_idx = 1 : nv        
        if nv_idx <= size(Mn, 2)
            %missing_ratio > 0
            cols = abs(Mn{nv_idx} - 1) < 1e-6;
            if  length(find(cols > 0)) < num_sample
                X = data_views{nv_idx}(:, cols);
                W = constructW_PKN(X, kn);
                weights = zeros(num_sample, num_sample);
                weights(cols, cols) = W;
            else
                %missing_ratio = 0
                weights = constructW_PKN(data_views{nv_idx}, kn);
            end
        else
            % concatenation
            weights = constructW_PKN(data_views{nv_idx}, kn);
        end
        Z_views{nv_idx} = weights;

        D = diag(1./sqrt(sum(weights, 2)+ eps));
        W = D * weights * D;
        [U, ~, ~] = svd(W);
        V = U(:, 1 : num_clusters);       
        VV = normr(V);
        H_views{nv_idx} = VV;

        Hs = Hs + VV;

    end

    rand('state', 1000);
    labels = kmeans(Hs, num_clusters, 'maxiter', 1000, 'replicates', 20, 'emptyaction', 'singleton');
    A = sparse(1:num_sample, labels, 1);
    Y = full(A);

    