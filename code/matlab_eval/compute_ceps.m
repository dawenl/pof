clear all

%rir = 'lecture';
%dep = '_dep';
dep = '';

%data_dir = strcat('reverb_', rir);
data_dir = strcat('../iPhone');

n_spk = 6;
n_sent = 1;

ceps_rev_mean = zeros(n_spk, n_sent);
ceps_rev_median = zeros(n_spk, n_sent);

ceps_dr_em_np_mean = zeros(n_spk, n_sent);
ceps_dr_em_np_median = zeros(n_spk, n_sent);

ceps_dr_me_np_mean = zeros(n_spk, n_sent);
ceps_dr_me_np_median = zeros(n_spk, n_sent);

ceps_dr_em_op_mean = zeros(n_spk, n_sent);
ceps_dr_em_op_median = zeros(n_spk, n_sent);

ceps_dr_me_op_mean = zeros(n_spk, n_sent);
ceps_dr_me_op_median = zeros(n_spk, n_sent);

ceps_dr_emp_mean = zeros(n_spk, n_sent);
ceps_dr_emp_median = zeros(n_spk, n_sent);

ceps_cmn_mean = zeros(n_spk, n_sent);
ceps_cmn_median = zeros(n_spk, n_sent);

ds_rev = [];
ds_dr_em_np = [];
ds_dr_me_np = [];
ds_dr_emp = [];
ds_dr_cnm = [];

param = struct('frame', 0.064, 'shift', 0.016, 'window', @hanning, ...
    'order', 24, 'timdif', 0.0, 'cmn', 'n');

fprintf('************Cepstrum distance**********\n');
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
        
%         file_dr_em_op = strcat(data_dir, '/spk', int2str(i), '_sent', ...
%             int2str(j), '_dr_op_em.wav');
%         file_dr_me_op = strcat(data_dir, '/spk', int2str(i), '_sent', ...
%             int2str(j), '_dr_op_me.wav');
        
        file_dr_emp = strcat(data_dir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_emp.wav');
        
        file_cmn = strcat(data_dir, dep, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_cmn.wav');
        
        [wav, fs] = wavread(file_org);
        [wav_rev, ~] = wavread(file_rev);
        
        [wav_dr_em_np, ~] = wavread(file_dr_em_np);
        [wav_dr_me_np, ~] = wavread(file_dr_me_np);
        
%         [wav_dr_em_op, ~] = wavread(file_dr_em_op);
%         [wav_dr_me_op, ~] = wavread(file_dr_me_op);
        
        [wav_dr_emp, ~] = wavread(file_dr_emp);
        [wav_cmn, ~] = wavread(file_cmn);
        
        [ceps_rev_mean(i, j), ceps_rev_median(i, j), tmp] = cepsdist(wav, wav_rev, fs, param);
        ds_rev = [ds_rev, tmp];
        [ceps_dr_em_np_mean(i, j), ceps_dr_em_np_median(i, j), tmp] = cepsdist(wav, wav_dr_em_np, fs, param);
        ds_dr_em_np = [ds_dr_em_np, tmp];
        %[ceps_dr_em_op_mean(i, j), ceps_dr_em_op_median(i, j)] = cepsdist(wav, wav_dr_em_op, fs, param);
        
        [ceps_dr_me_np_mean(i, j), ceps_dr_me_np_median(i, j), tmp] = cepsdist(wav, wav_dr_me_np, fs, param);
        ds_dr_me_np = [ds_dr_me_np, tmp];
        %[ceps_dr_me_op_mean(i, j), ceps_dr_me_op_median(i, j)] = cepsdist(wav, wav_dr_me_op, fs, param);
        
        [ceps_dr_emp_mean(i, j), ceps_dr_emp_median(i, j), tmp] = cepsdist(wav, wav_dr_emp, fs, param);
        ds_dr_emp = [ds_dr_emp, tmp];
        [ceps_cmn_mean(i, j), ceps_cmn_median(i, j), tmp] = cepsdist(wav, wav_cmn, fs, param);
        ds_dr_cnm = [ds_dr_cnm, tmp];
        
        fprintf('********Speaker %d Sentence %d:*********\n', i, j);
        fprintf('Reverberant speech\tMean/Median: %.2f / %.2f\n',ceps_rev_mean(i, j), ceps_rev_median(i, j)); 
        fprintf('Dereverb with CMN\tMean/Median: %.2f / %.2f\n', ceps_cmn_mean(i, j), ceps_cmn_median(i, j)); 

        fprintf('Dereverb with noisy phase (EM)\tMean/Median: %.2f / %.2f\n', ceps_dr_em_np_mean(i, j), ceps_dr_em_np_median(i, j)); 
        fprintf('Dereverb with noisy phase (ME)\tMean/Median: %.2f / %.2f\n', ceps_dr_me_np_mean(i, j), ceps_dr_me_np_median(i, j)); 

        %fprintf('Dereverb with oracle phase (EM)\tMean/Median: %.2f / %.2f\n', ceps_dr_em_op_mean(i, j), ceps_dr_em_op_median(i, j)); 
        %fprintf('Dereverb with oracle phase (ME)\tMean/Median: %.2f / %.2f\n', ceps_dr_me_op_mean(i, j), ceps_dr_me_op_median(i, j)); 

        fprintf('Dereverb with emp filter\tMean/Median: %.2f / %.2f\n\n', ceps_dr_emp_mean(i, j), ceps_dr_emp_median(i, j)); 
    end
end

n = length(ds_rev);
fprintf('*********On average********\n')
fprintf('Reverberant speech\t%.2f / %.2f / %.2f\n', mean(ds_rev), median(ds_rev), std(ds_rev)/sqrt(n));
fprintf('Dereverb with CMN\t%.2f / %.2f / %.2f\n', mean(ds_dr_cnm), median(ds_dr_cnm), std(ds_dr_cnm)/sqrt(n));
fprintf('Dereverb with noisy phase (EM)\t%.2f / %.2f / %.2f\n', mean(ds_dr_em_np), median(ds_dr_em_np), std(ds_dr_em_np)/sqrt(n));
fprintf('Dereverb with noisy phase (ME)\t%.2f / %.2f / %.2f\n', mean(ds_dr_me_np), median(ds_dr_me_np), std(ds_dr_me_np)/sqrt(n));
fprintf('Dereverb with emp filter\t%.2f / %.2f / %.2f\n', mean(ds_dr_emp), median(ds_dr_emp), std(ds_dr_emp)/sqrt(n));

fprintf('p-value = %.3f\n', signrank(ceps_dr_me_np_mean, ceps_cmn_mean));
fprintf('p-value = %.3f\n', signrank(ceps_dr_me_np_mean, ceps_dr_emp_mean));
fprintf('p-value = %.3f\n', signrank(ceps_cmn_mean, ceps_dr_emp_mean));