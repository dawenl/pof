clear all

rir = 'lecture';

n_spk = 6;
n_sent = 1;

lpcllr_rev_mean = zeros(n_spk, n_sent);
lpcllr_rev_median = zeros(n_spk, n_sent);

lpcllr_dr_em_np_mean = zeros(n_spk, n_sent);
lpcllr_dr_em_np_median = zeros(n_spk, n_sent);

lpcllr_dr_me_np_mean = zeros(n_spk, n_sent);
lpcllr_dr_me_np_median = zeros(n_spk, n_sent);

lpcllr_dr_em_op_mean = zeros(n_spk, n_sent);
lpcllr_dr_em_op_median = zeros(n_spk, n_sent);

lpcllr_dr_me_op_mean = zeros(n_spk, n_sent);
lpcllr_dr_me_op_median = zeros(n_spk, n_sent);

lpcllr_dr_emp_mean = zeros(n_spk, n_sent);
lpcllr_dr_emp_median = zeros(n_spk, n_sent);

param = struct('frame', 0.064, 'shift', 0.016, 'window', @hanning, ...
    'lpcorder', 12);

fprintf('************LPCLLR**********\n');
for i = 1:n_spk
    for j = 1:n_sent
        file_org = strcat('reverb_', rir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_org.wav');
        file_rev = strcat('reverb_', rir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_rev.wav');
        
        file_dr_em_np = strcat('reverb_', rir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_np_em.wav');
        file_dr_me_np = strcat('reverb_', rir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_np_me.wav');
        
        file_dr_em_op = strcat('reverb_', rir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_op_em.wav');
        file_dr_me_op = strcat('reverb_', rir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_op_me.wav');
        
        file_dr_emp = strcat('reverb_', rir, '/spk', int2str(i), '_sent', ...
            int2str(j), '_dr_emp.wav');
        
        [wav, fs] = wavread(file_org);
        [wav_rev, ~] = wavread(file_rev);
        
        [wav_dr_em_np, ~] = wavread(file_dr_em_np);
        [wav_dr_me_np, ~] = wavread(file_dr_me_np);
        
        [wav_dr_em_op, ~] = wavread(file_dr_em_op);
        [wav_dr_me_op, ~] = wavread(file_dr_me_op);
        
        [wav_dr_emp, ~] = wavread(file_dr_emp);
        
        [lpcllr_rev_mean(i, j), lpcllr_rev_median(i, j)] = lpcllr(wav_rev, wav, fs, param);
        [lpcllr_dr_em_np_mean(i, j), lpcllr_dr_em_np_median(i, j)] = lpcllr(wav_dr_em_np, wav, fs, param);
        [lpcllr_dr_em_op_mean(i, j), lpcllr_dr_em_op_median(i, j)] = lpcllr(wav_dr_em_op, wav, fs, param);
        
        [lpcllr_dr_me_np_mean(i, j), lpcllr_dr_me_np_median(i, j)] = lpcllr(wav_dr_me_np, wav, fs, param);
        [lpcllr_dr_me_op_mean(i, j), lpcllr_dr_me_op_median(i, j)] = lpcllr(wav_dr_me_op, wav, fs, param);
        
        [lpcllr_dr_emp_mean(i, j), lpcllr_dr_emp_median(i, j)] = lpcllr(wav_dr_emp, wav, fs, param);
        
        fprintf('********Speaker %d Sentence %d:*********\n', i, j);
        fprintf('Reverberant speech\tMean/Median: %.2f / %.2f\n',lpcllr_rev_mean(i, j), lpcllr_rev_median(i, j)); 
        
        fprintf('Dereverb with noisy phase (EM)\tMean/Median: %.2f / %.2f\n', lpcllr_dr_em_np_mean(i, j), lpcllr_dr_em_np_median(i, j)); 
        fprintf('Dereverb with noisy phase (ME)\tMean/Median: %.2f / %.2f\n', lpcllr_dr_me_np_mean(i, j), lpcllr_dr_me_np_median(i, j)); 

        fprintf('Dereverb with oracle phase (EM)\tMean/Median: %.2f / %.2f\n', lpcllr_dr_em_op_mean(i, j), lpcllr_dr_em_op_median(i, j)); 
        fprintf('Dereverb with oracle phase (ME)\tMean/Median: %.2f / %.2f\n', lpcllr_dr_me_op_mean(i, j), lpcllr_dr_me_op_median(i, j)); 

        fprintf('Dereverb with emp filter\tMean/Median: %.2f / %.2f\n\n', lpcllr_dr_emp_mean(i, j), lpcllr_dr_emp_median(i, j)); 
    end
end

        
        