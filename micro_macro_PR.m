function [ micro, macro] = micro_macro_PR( pred_label , orig_label)
%computer micro and macro: precision, recall and fscore
%Sandy wltongxing@163.com
%micro>macro?

mat=confusionmat(orig_label, pred_label);
%label_unique=unique([orig_label(:);pred_label(:)]);
%     microTP=0;
%     microFP=0;
%     microFN=0;
len=size(mat,1);
macroTP=zeros(len,1);
macroFP=zeros(len,1);
macroFN=zeros(len,1);
macroP=zeros(len,1);
macroR=zeros(len,1);
macroF=zeros(len,1);

for i=1:len
    macroTP(i)=mat(i,i);
    macroFP(i)=sum(mat(:, i))-mat(i,i);
    macroFN(i)=sum(mat(i,:))-mat(i,i);
    macroP(i)=macroTP(i)/(macroTP(i)+macroFP(i));

    if isnan(macroP(i))
        macroP(i) = 0;
    end
    
    macroR(i)=macroTP(i)/(macroTP(i)+macroFN(i));
    if isnan(macroR(i))
        macroR(i) = 0;
    end
    

    denom = macroP(i)+macroR(i);
    if denom == 0
        macroF(i) = 0; 
    else
        macroF(i)=2*macroP(i)*macroR(i)/denom;
    end
end

macro.precision=mean(macroP);
macro.recall=mean(macroR);
macro.fscore=mean(macroF);

micro.precision=sum(macroTP)/(sum(macroTP)+sum(macroFP));
if isnan(micro.precision), micro.precision = 0; end

micro.recall=sum(macroTP)/(sum(macroTP)+sum(macroFN));
if isnan(micro.recall), micro.recall = 0; end

micro.fscore=2*micro.precision*micro.recall/(micro.precision+micro.recall);
if isnan(micro.fscore), micro.fscore = 0; end

end