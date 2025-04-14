function [Y_star, iter, obj_values] = oagl(Zn, Hn, Y, alpha, beta)

mu = 1e-4;
mu_max = 1e6;
rho = 1.2;
iter = 0;
tol = 1e-6;
maxIter = 200;
obj_values = zeros(1, maxIter);

max_iter2 = 3;
fobj2 = zeros(max_iter2, 1);

nv = length(Hn);
[num, k] = size(Hn{1});

lambdas = ones(nv, 1) * sqrt(1 / nv);
Fn_results = zeros(nv, 1);

R = zeros(num, num, nv);
G = zeros(num, num, nv);

Hv = zeros(num, k, nv);
Rv = zeros(k, k, nv);
% Bv = zeros(num, num, nv);

%Initialization
for nv_idx = 1 : nv
   Hv(:, :, nv_idx) = Hn{nv_idx};
   Rv(:, :, nv_idx) = eye(k);
end
% max_eigen_values = zeros(1, nv);

while iter < maxIter
    iter = iter + 1;

    % update Y
    Q = zeros(num, k);
    for nv_idx = 1 : nv
        Q = Q + lambdas(nv_idx) * Hv(:, :, nv_idx) * Rv(:, :, nv_idx);    
    end
    Y = update_Y(Y, Q, num, k);
    Y_star = Y * (Y' * Y + eps * eye(k))^(-0.5);

    % update Rv
    for nv_idx = 1 : nv
        M = Hv(:, :, nv_idx)' * Y_star;
        [Um, ~, Vm] = svd(M, 'econ');
        Rv(:, :, nv_idx) = Um * Vm';
    end
    
    % update Hv
    A = G - 1 / mu * R;
    for nv_idx = 1 : nv
        iter2 = 0;
        B = alpha * Zn{nv_idx} + mu / 2 * (A(:, :, nv_idx) + A(:, :, nv_idx)');

        %cache used
%         cache = false;
%         if nv_idx > 1
%             result = max(max(abs(Bv(:, :, nv_idx-1)-B)));
%             if result < 0.01
%                 max_eigen = max_eigen_values(nv_idx-1);
%                 cache = true;
%             end
%         end
%         if ~cache
%             if num > 5000
%                 [~, d] = eig(B);
%                 d = diag(d);
%                 [d1, ~] = sort(d,'descend');
%                 max_eigen = d1(1);
%             else
%                 max_eigen = eigs(B, 1 ,'la');
%             end    
%             max_eigen_values(nv_idx) = max_eigen;
% %             disp(max_eigen);
%         end
%         Bv(:, :, nv_idx) = B;

        %cache abandoned
%         max_eigen = eigs(L_star, 1 ,'la');

        while iter2 < max_iter2
            iter2 = iter2 + 1;   
            M = B * Hv(:, :, nv_idx) + beta * lambdas(nv_idx) * Y_star * Rv(:, : , nv_idx)';
            [Um, ~, Vm] = svd(M, 'econ');
            Hv(:, :, nv_idx) = Um * Vm';
            fobj2(iter2) = trace(Hv(:, :, nv_idx)' * M);
            if iter2 > 1
                r = (fobj2(iter2)-fobj2(iter2-1))/fobj2(iter2);
%                 disp([iter2, fobj2(iter2), r]);  
                if r < 0.1
                    break;
                end
%             else
%                 disp([iter2, fobj2(iter2)]); 
            end          
        end        
    end

     % update G
    for nv_idx = 1 : nv
        G( : , : , nv_idx) = Hv(:, :, nv_idx) * Hv(:, :, nv_idx)';
    end
    tempG1 = G + 1 / mu * R;
    tempG = shiftdim(tempG1, 1);
    [tempT, ~, ~] = prox_tnn(tempG, 1 / mu);
    T = shiftdim(tempT, 2);        
    
    % update lambda
    for nv_idx = 1 : nv
        Fn_results(nv_idx) = trace(Hv(:, :, nv_idx)' * Y_star * Rv(:, : , nv_idx)');
    end
    lambdas = Fn_results  / norm(Fn_results);


     %update R and rho
    R = R + mu * (G - T);    
    mu = min(mu * rho, mu_max);
        
    %calculate the error
    err = max(max(max(abs(G - T))));
    obj_values(1, iter) = err;
%     disp([iter, err]);
    if err < tol
%         disp([iter, err]);
        break;
    end

end

end

