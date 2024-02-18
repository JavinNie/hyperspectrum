%% function test setting
% encoding_matrix=rand(250,30);
% lamuda=linspace(1200,1700,250);
% draw_flag=1;

function Matrix_evaluation_indice=matrix_evaluation(encoding_matrix,lamuda, draw_flag)

%wavelength_samples:wavelength samples number,channels:channel number

[wavelength_samples,channels]=size(encoding_matrix);


%% evaluation 1.1. 自相关半峰半宽 auto-correlation half-width-half-maximum (HWHM)
xcorr_coeff=xcorr(encoding_matrix,'normalized');
acf=mean(xcorr_coeff(:,[1:channels+1:end]),2);% Only the autocorrelation of the index of the diagonal line is extracted
acf=acf(ceil(length(acf)/2):end);
index_cross_point=find_FWHM_point(acf);
delta_lamuda=lamuda-min(lamuda);
lamuda_index_HWHM=interp1((1:length(delta_lamuda)),delta_lamuda,index_cross_point);
fprintf('The Auto-Correlation Half-Width-Half-Maximum (HWHM) of the filter matrix is %.2fnm\n',lamuda_index_HWHM)

%% draw
if draw_flag==1
   plot(delta_lamuda,acf,'-');%, 'LineWidth', 2, 'Color', '#0072BD'); % 蓝色
   xlabel('Δλ/nm')
   ylabel('C(Δλ)')
   title('Auto-correlation function curve of the encoding matrix')
   hold on
   plot(lamuda_index_HWHM,0.5,'*','linewidth',5)
   plot(linspace(0,lamuda_index_HWHM,100),0.5*max(acf)*ones(1,100),'r-')
   plot(lamuda_index_HWHM*ones(1,100),linspace(0,0.5,100),'r-')
   text(lamuda_index_HWHM,0.54,sprintf('δλ=%.2fnm',lamuda_index_HWHM));
   hold off
   exportgraphics(gca, 'Auto-correlation function curve.tif', 'Resolution', 300);
end
acf_HWHM=lamuda_index_HWHM;

%% evaluation 1.2. 互相关系数cross-correlation coefficient
cce_matrix = corrcoef(encoding_matrix);
cce_matrix(1:channels+1:end)=[];%clear diagonal
cce=mean(cce_matrix,'all');
fprintf('The Average Cross-correlation Coefficient of the filter matrix is %.2f\n',cce)

%% evaluation 1.3. 矩阵条件数condition number
cond_number=cond(encoding_matrix);
fprintf('The Condition Number of the filter matrix is %.2f\n',cond_number)

% %% evaluation 1.4. 矩阵 标准差/均值
% cond_number=cond(encoding_matrix);
% fprintf('The Condition Number of the filter matrix is %.2f\n',cond_number)

%% conclusion
Matrix_evaluation_indice.acf_HWHM=acf_HWHM;
Matrix_evaluation_indice.cross_correlation=cce;
Matrix_evaluation_indice.cond_number=cond_number;

end

















