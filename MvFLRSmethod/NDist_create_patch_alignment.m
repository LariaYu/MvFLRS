function [alignment, W]= NDist_create_patch_alignment(views,mvParam)
%CREATE_PATCH_ALIGNMENT - Creates 3D matrix composed of 2D neighborhood distance matrices.
%
% Syntax:  alignment = create_patch_alignment(views,k,methods)
%
% Inputs:
%    views - Cell array of feature matrices
%    k - Neighborhood size
%    t - Kernel parameter 
%    methods - Method for each view to be used in the distance calculations
%
% Outputs:
%    alignment - Patch alignment
%
% Example: 
%    create_patch_alignment({view1,view2},30,{'euclidean','euclidean']);
%
% Author: Robert Ciszek
% email: ciszek@uef.fi
% August 2014; Last revision: 21-August-2016
    t=mvParam.t;
    k=mvParam.k;
%     methods=mvParam.methods;
    n = size(views{1,1},1);
    v = size(views,2);
    L = zeros(n,n,v);
    W = zeros(n,n,v); 
    D = zeros(n,n,v); 
    for v_i=1:v
        [Dist] = Matrix_Distance_wEu(views{1,v_i});
        distances = squareform(Dist);   %.^(1/delta);   % .^2;  %ʵ���֪������ԭ�о���Ч���Ϻ�
%         distances = squareform(pdist(views{1,v_i},char(methods(v_i)))).^2; 
        %pdist����ѡ��ϵͳ���빫ʽ��Ҳ����ѡȡ�Զ�����빫ʽ�������ڴ˴����иĽ���ʵ�ִ��£�
        distances(isinf(distances)) = 0;
        distances(isnan(distances)) = 0;
        distances  = exp(-distances/t);   %��˹�˺���,��һ�����뵽��0,1��
%%     �ھ���������������ϵ��
%         cf=exp(-corrcoef(distances));
%         distances=distances.*cf;
	    [~, indexes] = sort(distances,2,'descend');  % �������У�������С�෴
        fprintf('distance_view %d',v_i);
%% k-nearest neighbors,ÿ��ʾ����K�������С��Ȩ�ر�����W������,Ȩ�غ;����С�෴
        for p=1:n   
            W(p,indexes(p,2:2+k-1),v_i) = distances(p,indexes(p,2:2+k-1));
                    W(indexes(p,2:2+k-1),p,v_i) = distances(p,indexes(p,2:2+k-1));	% k-dimensional column vector weighted	  
        end
        fprintf('W_view %d',v_i);
%         W=W.*cf;
%% ���Ȩ�ؾ���Ĺ���������������˹����L=D-W,���㷨ʹ�ù�һ����ʾL=I-D^(-1/2)*W*D^(-1/2)
        W(1+(n^2)*(v_i-1):n+1:v_i*(n^2)) = 0;        
	    D(1+(n^2)*(v_i-1):n+1:v_i*(n^2)) = sum(W(:,:,v_i),2);
        L(:,:,v_i) = eye(n) - D(:,:,v_i)^(-1/2)*W(:,:,v_i)*D(:,:,v_i)^(-1/2);     % Eqn��11��, a normalized graph Laplacian matrix
        fprintf('L_view %d\n',v_i);
    end    
    alignment = L;  
end
