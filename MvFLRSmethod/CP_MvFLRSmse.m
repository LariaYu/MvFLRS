function [Y, alpha] = CP_MvFLRSmse(views,mv_param)
%MSE - Returns a multiview spectral embedding of the data given as a parameter
%A MATLAB implementation of the Multiview Spectral Embedding method proposed by T. Xia, D. Tao, 
%T. Mei and Y. Zhang in "Multiview Spectral Embedding," in IEEE Transactions on Systems, 
%Man, and Cybernetics, Part B (Cybernetics), vol. 40, no. 6, pp. 1438-1446, Dec. 2010.
%
% Syntax:  [reduction] = mse(views,d,k,r,t,max_iter, methods)
%
% Inputs:
%    views - A cell array of NxM matrices
%    d - Final dimensionality
%    k - Neighborhood size
%    r - Regularization factor, r > 1
%    t - kernel parameter
%    max_iter - Maximum number of iterations
%    methods - A cell array of methods for distance evaluation.
%	       All methods supported by pdist are allowed. 
%
% Outputs:
%    reduction - Multiview dimensionality reduction
%
% Example: 
%    mse({first_view, second_view},2,30,3,100,{'euclidean','jaccard'})
%
% Other m-files required: create_patch_alignment.m
%
% Author: Robert Ciszek 
% August 2014; Last revision: 21-August-2016
%参数传递

d=mv_param.d;  %    d - Final dimensionality
k=mv_param.k;  %    k - Neighborhood size  35
r=mv_param.r;   %    r - Regularization factor, r > 1,best values 5<r<9
v=mv_param.v;   %    v-numbers of views
t=mv_param.t;  %     t - kernel parameter
max_iter=mv_param.max_iter; %    max_iter - Maximum number of iterations

% 每个视角对应一个距离度量公式
    if ~exist('methods', 'var')
        methods = repmat({'euclidean'},1,size(views,2));
    end

    n = size(views{1,1},1);
    m = size(views,2);
    L = NDist_create_patch_alignment(views,mv_param);  %a normalized graph Laplacian matrix
    alpha = ones(1,m) / m;                            %这里的alpha为平均值，是否可以考虑取不同的值？
    previous_alpha = alpha; 
    Y = zeros(d,n);                                  %最终获取的特征表示
    previous_Y = ones(d,n);
    iteration = 1;
    ak=0;
    %------------------------CP------------------------
    %张量CP分解--重构
    [ll,~,lens]=size(L);
    T=zeros(ll,d,lens);
    for il=1:lens
        [V,D] = eig(L(:,:,il));           %Eigenvalues and eigenvectors
        e = diag(D);        
        [~, I] = sort(e);
        T(:,:,il) = transpose(V(:,I(2:d+1)))'; 
    end
    [Ws]=W_CP_tensor(T);     
    alphaV=exp(Ws.V.^2);
    alphaV=alphaV/sum(alphaV(:));   
    %------------------------CP------------------------    
    while previous_Y ~= Y
        
        fprintf('Iteration: %i\n', iteration);
        iteration = iteration +1; 
        previous_Y = Y; 
	%Calculate weighted L
	   w_L =  sum(bsxfun(@times,L,reshape(alpha,[1 1 m])),3);    %融合三种特征,加权相加
	         %  @times          Array multiply
%        w_L= w_L.*abs(MatrixCP);   %不好，效果  
%         w_L= w_L.*MatrixCP;
	   [V,D] = eig(w_L);           %Eigenvalues and eigenvectors
     %addational files 
%         f_w_L = w_L + eye(size(w_L))*(-2*min(eig(w_L)));  %Enq(14)?
%         [V,D] = eig(f_w_L); 
        
	    e = diag(D);
        
        [B, I] = sort(e);
        Y = transpose(V(:,I(2:d+1)));                  %非共轭转置,I(2:d+1)为什么从第2个特征值开始取元素？

        disp(strcat('Alpha:',sprintf(' %f',alpha ))); 
        
% 	    disp(sprintf('Alpha: %f ',alpha ));
        if iteration >= max_iter
           break; 
        end
        
        if sum(previous_alpha-alpha)<0.0000001
            ak=ak+1;
        end
        previous_alpha = alpha; 
        if ak>=4
            break;
        end
        alpha = calculate_alpha( Y, L, m, r);            %新增加变化,更新alpha的值
        alpha=alpha+alphaV;
        alpha=alpha/sum(alpha(:));
    end
    
    function alpha = calculate_alpha( Y, L, m, r )
        %  Global Coordinate Alignment， Eqn(16)
        alpha = ones(1,m);
        %优化运算
        for m_i = 1:m
           alpha(1,m_i) = (1/trace(Y*L(:,:,m_i)*transpose(Y)))^(1/(r-1)); 
        end
            alpha=alpha/sum(alpha(:));
    end

end

