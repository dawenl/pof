clear all

%rir = 'office';
%dep = '_dep';
dep = '';

%data_dir = strcat('reverb_', rir);
data_dir = strcat('../iPhone');

n_spk = 6;
n_sent = 1;

srmr_rev = zeros(n_spk, n_sent);

srmr_dr_em_np = zeros(n_spk, n_sent);

srmr_dr_me_np = zeros(n_spk, n_sent);

srmr_dr_emp = zeros(n_spk, n_sent);

srmr_cmn = zeros(n_spk, n_sent);


fprintf('\n************SRMR**********\n');
for i = 1:n_spk
    for j = 1:n_sent
        file_rev = strcat(data_dir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_rev.wav');
        
        file_dr_em_np = strcat(data_dir, dep, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_np_em.wav');
        file_dr_me_np = strcat(data_dir, dep, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_np_me.wav');
        
        file_dr_emp = strcat(data_dir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_emp.wav');
        
        file_cmn = strcat(data_dir, dep, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_cmn.wav');
       
        srmr_rev(i, j) = SRMR_main(file_rev);
        srmr_cmn(i, j) = SRMR_main(file_cmn);
        srmr_dr_em_np(i, j) = SRMR_main(file_dr_em_np);
        srmr_dr_me_np(i, j) = SRMR_main(file_dr_me_np);
        srmr_dr_emp(i, j) = SRMR_main(file_dr_emp);
        
        fprintf('********Speaker %d Sentence %d:*********\n', i, j);
        fprintf('Reverberant speech: %.2f\n',srmr_rev(i, j)); 
        fprintf('Dereverb with CMN: %.2f\n', srmr_cmn(i, j)); 

        fprintf('Dereverb with noisy phase (EM): %.2f\n', srmr_dr_em_np(i, j)); 
        fprintf('Dereverb with noisy phase (ME): %.2f\n', srmr_dr_me_np(i, j)); 

        fprintf('Dereverb with emp filter: %.2f\n\n', srmr_dr_emp(i, j)); 
    end
end

n = length(srmr_rev);
fprintf('*********On average********\n')
fprintf('Reverberant speech\t%.2f / %.2f / %.2f\n', mean(srmr_rev), median(srmr_rev), std(srmr_rev)/sqrt(n));
fprintf('Dereverb with CMN\t%.2f / %.2f / %.2f\n', mean(srmr_cmn), median(srmr_cmn), std(srmr_cmn)/sqrt(n));
fprintf('Dereverb with noisy phase (EM)\t%.2f / %.2f / %.2f\n', mean(srmr_dr_em_np), median(srmr_dr_em_np), std(srmr_dr_em_np)/sqrt(n));
fprintf('Dereverb with noisy phase (ME)\t%.2f / %.2f / %.2f\n', mean(srmr_dr_me_np), median(srmr_dr_me_np), std(srmr_dr_me_np)/sqrt(n));
fprintf('Dereverb with emp filter\t%.2f / %.2f / %.2f\n', mean(srmr_dr_emp), median(srmr_dr_emp), std(srmr_dr_emp)/sqrt(n));

fprintf('p-value = %.4f\n', signrank(srmr_dr_me_np, srmr_cmn));
fprintf('p-value = %.4f\n', signrank(srmr_dr_me_np, srmr_dr_emp));
fprintf('p-value = %.4f\n', signrank(srmr_cmn, srmr_dr_emp));

        