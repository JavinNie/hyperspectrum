function [index_cross_point]=find_FWHM_point(curve)
%%find cross-points between a curve and aa horizontal line with static number 
threshold_number=0.5*max(curve);
index_temp=find(curve>threshold_number);
index1=index_temp(end);
index2=index_temp(end)+1;
index_cross_point=interp1([curve(index1),curve(index2)],[index1,index2],threshold_number);

end