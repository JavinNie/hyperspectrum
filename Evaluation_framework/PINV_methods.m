function sig_sim=PINV_methods(A,B)
%输入A：[nfilter,samples]
%输入B：[nfilter,1]
%输出sig_sim：[samples,1]
    sig_sim=pinv(A)*B;
%     sig_sim=kalman_filter(sig_sim);
end
