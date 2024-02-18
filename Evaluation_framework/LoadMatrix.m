function encoding_matrix=LoadMatrix(file_adress)
%% load
encoding_matrix=load(file_adress);

[RR,CC]=size(encoding_matrix);
formatspec='The encoding matrix''s wavelength samples number is:%d, channel number is:%d.\n';
fprintf(formatspec,RR,CC);%output the size of filter matrix

% %% check and correct
% for ii=1:10
% [RR,CC]=size(encoding_matrix);
% formatspec='The encoding matrix''s wavelength samples number is:%d, channel number is:%d.\n';
% fprintf(formatspec,RR,CC);%output the size of filter matrix
% Trans_flag=input('Do I need to transpose it or just TERMINATE this program?([Y/N/T])','s');
% if Trans_flag=='Y'
%     encoding_matrix=encoding_matrix';
%     continue
% elseif Trans_flag=='N'
%     break
% elseif Trans_flag=='T'
%     error('wrong status in loadingmatrix')
% end
% 
% disp("Terminating due to too many input errors")
%     
end