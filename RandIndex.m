function [RI] = RandIndex(L_opt,L_true)

    [mLo nLo]=size(L_opt);
    [mLt nLt]=size(L_true);
    TP=0;TN=0;FN=0;FP=0;
    for i=1:nLo
        for j=i+1:nLo
            n1=sum(L_opt(:,i)==L_opt(:,j));
            n2=sum(L_true(:,i)==L_true(:,j));
            if ((n1==mLo)&&(n2==mLt))
                TP=TP+1;
            elseif((n1<mLo)&&(n2<mLt))
                TN=TN+1;
            elseif((n1==mLo)&&(n2<mLt))
                FN=FN+1;
            else
                FP=FP+1;
            end
        end
    end
    
    RI = (TP+TN)/(TP+TN+FN+FP);
    end
