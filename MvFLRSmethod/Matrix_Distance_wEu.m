function [Dist] = Matrix_Distance_wEu(HistMatrix,mode)
     [ImageNums,~]=size(HistMatrix);  
        if nargin~=2
            mode=1;
        end
        Dist=[];
        switch mode
            case 1
                for i=1:ImageNums-1
                MyHist=HistMatrix(i,:);
                TmpHistMatrix=repmat(MyHist,ImageNums-i,1);
                LastHistMatix=HistMatrix(i+1:ImageNums,:);
                TmpDist=abs(TmpHistMatrix-LastHistMatix)./(1+TmpHistMatrix+LastHistMatix);        
                Dist= [Dist sum(TmpDist,2)'];
                end
            case 2                
                Dist=pdist2(MyHist,HistMatrix,'cityblock');
            case 3
                lens=size(MyHist,1);
                for i=1:lens
                    TmpHistMatrix=repmat(MyHist(i,:),ImageNums,1);
                    TmpDist=abs(TmpHistMatrix-HistMatrix)./(1+TmpHistMatrix+HistMatrix);        
                    Dist(i,:)=sum(TmpDist,2)';
                end
        end
        Dist=(round(Dist.*10000))./10000;
end

