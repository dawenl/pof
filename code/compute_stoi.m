%clear all

n_st = 60;

stoi_sf = zeros(1, n_st);

%stoi_kl = zeros(1, n_st);

%stoi_cutoff = zeros(1, n_st);

for i = 1:n_st
    file_org = strcat('bwe/', int2str(i), '_org.wav');
    %file_cutoff = strcat('bwe/', int2str(i), '_cutoff.wav');
    %file_kl_rec = strcat('bwe/', int2str(i), '_kl_rec.wav');
    file_sf_rec = strcat('bwe/', int2str(i), '_sf_rec.wav');


    [x, fs] = wavread(file_org);
    
    %[y, ~] = wavread(file_cutoff);
    %stoi_cutoff(i) = stoi(x, y, fs);

    %[y, ~] = wavread(file_kl_rec);
    %stoi_kl(i) = stoi(x, y, fs);

    [y, ~] = wavread(file_sf_rec);
    stoi_sf(i) = stoi(x, y, fs);
end
        
        