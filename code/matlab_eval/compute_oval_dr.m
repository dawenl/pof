clear all

%rir = 'office';
%dep = '_dep';
dep = '';

%data_dir = strcat('reverb_', rir);
data_dir = strcat('../iPhone');

n_spk = 6;
n_sent = 1;

ovrl_rev = zeros(n_spk, n_sent);

ovrl_dr_em_np = zeros(n_spk, n_sent);

ovrl_dr_me_np = zeros(n_spk, n_sent);

ovrl_dr_emp = zeros(n_spk, n_sent);

ovrl_cmn = zeros(n_spk, n_sent);


fprintf('\n************OVRL**********\n');
for i = 1:n_spk
    for j = 1:n_sent
        file_org = strcat(data_dir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_org.wav');
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
       
        [ovrl_rev(i, j), ~] = composite(file_org, file_rev);
        [ovrl_cmn(i, j), ~] = composite(file_org, file_cmn);
        [ovrl_dr_em_np(i, j), ~] = composite(file_org, file_dr_em_np);
        [ovrl_dr_me_np(i, j), ~] = composite(file_org, file_dr_me_np);
        [ovrl_dr_emp(i, j), ~] = composite(file_org, file_dr_emp);
        
        fprintf('********Speaker %d Sentence %d:*********\n', i, j);
        fprintf('Reverberant speech: %.2f\n',ovrl_rev(i, j)); 
        fprintf('Dereverb with CMN: %.2f\n', ovrl_cmn(i, j)); 

        fprintf('Dereverb with noisy phase (EM): %.2f\n', ovrl_dr_em_np(i, j)); 
        fprintf('Dereverb with noisy phase (ME): %.2f\n', ovrl_dr_me_np(i, j)); 

        fprintf('Dereverb with emp filter: %.2f\n\n', ovrl_dr_emp(i, j)); 
    end
end

n = length(ovrl_rev);
fprintf('*********On average********\n')
fprintf('Reverberant speech\t%.2f / %.2f / %.2f\n', mean(ovrl_rev), median(ovrl_rev), std(ovrl_rev)/sqrt(n));
fprintf('Dereverb with CMN\t%.2f / %.2f / %.2f\n', mean(ovrl_cmn), median(ovrl_cmn), std(ovrl_cmn)/sqrt(n));
fprintf('Dereverb with noisy phase (EM)\t%.2f / %.2f / %.2f\n', mean(ovrl_dr_em_np), median(ovrl_dr_em_np), std(ovrl_dr_em_np)/sqrt(n));
fprintf('Dereverb with noisy phase (ME)\t%.2f / %.2f / %.2f\n', mean(ovrl_dr_me_np), median(ovrl_dr_me_np), std(ovrl_dr_me_np)/sqrt(n));
fprintf('Dereverb with emp filter\t%.2f / %.2f / %.2f\n', mean(ovrl_dr_emp), median(ovrl_dr_emp), std(ovrl_dr_emp)/sqrt(n));

fprintf('p-value = %.4f\n', signrank(ovrl_dr_me_np, ovrl_cmn));
        