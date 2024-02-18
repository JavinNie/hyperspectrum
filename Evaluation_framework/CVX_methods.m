function sig_sim=CVX_methods(A,B)
%     tic
% %     CVX
%     cvx_begin
%     variable x(m)
%     minimize( norm( B-A' * x, 2 )+0.00005*norm(x,2))      %目标函数
% %     minimize( norm( B-A' * x, 2 )) %目标函数
%     subject to
%         0 <= x <=1
%         C * x == d                                     %约束条件1
%         norm( x, Inf ) <= e                      %约束条件2
%     cvx_end    
%     sig_sim3=x;
%     toc
   
    
    
end
