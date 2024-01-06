function [normHist] = DR_HistNorm(hist)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    [m,n]=size(hist);
    maxs=max(hist'); %#ok<UDIM>
    mins=min(transpose(hist));
    ms=maxs-mins;
    rms=repmat(ms',1,n);
    hist=hist+rms;
    sm=sum(hist,2);
    rsm=repmat(sm,1,n);
    normHist=(hist./rsm); 
    normHist=normHist*n;
end

