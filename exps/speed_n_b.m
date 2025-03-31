% Script to test chop with half precision, multiple runs, and averaging
clear all;
clc;
pc = py.importlib.import_module('pychop');
np = py.importlib.import_module('numpy');
th = py.importlib.import_module('torch');
% torch.cuda.is_available()

addpath("../tests/chop")
pe = pyenv()
% Define matrix sizes
rng(0);

sizes = 2.^[8, 9, 10, 11, 12, 13]; % original: [128, 512, 2048, 8192, 32768] 

% Define rounding modes supported by chop
rounding_modes = [1, 2, 3, 4, 5, 6]; % 1: nearest (even), 2: up, 3: down, 4: zero, 5: stochastic (prop) 6. stochastic (uniform)
mode_names = {'Nearest (even)', 'Up', 'Down', 'Zero', 'Stochastic (prop)', 'Stochastic (uniform)'};

% Number of runs (11 total, discard first, average last 10)
num_runs = 11;

% Initialize runtime storage (for all runs) - matlab
runtimes_all = zeros(length(sizes), length(rounding_modes), num_runs);
% Initialize average runtime storage (after discarding first run)
runtimes_avg = zeros(length(sizes), length(rounding_modes));

% Initialize runtime storage (for all runs) for pychop (NumPy)
runtimes_all_np = zeros(length(sizes), length(rounding_modes), num_runs);
% Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_np = zeros(length(sizes), length(rounding_modes));

% Initialize runtime storage (for all runs) for pychop (NumPy)
runtimes_all_np2 = zeros(length(sizes), length(rounding_modes), num_runs);
% Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_np2 = zeros(length(sizes), length(rounding_modes));

% Initialize runtime storage (for all runs) for pychop (Torch)
runtimes_all_th = zeros(length(sizes), length(rounding_modes), num_runs);
% Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_th = zeros(length(sizes), length(rounding_modes));

% Initialize runtime storage (for all runs) for pychop (Torch)
runtimes_all_th2 = zeros(length(sizes), length(rounding_modes), num_runs);
% Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_th2 = zeros(length(sizes), length(rounding_modes));

% Initialize runtime storage (for all runs) for pychop (Torch GPU)
runtimes_all_th_gpu = zeros(length(sizes), length(rounding_modes), num_runs);
% Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_th_gpu = zeros(length(sizes), length(rounding_modes));

% Initialize runtime storage (for all runs) for pychop (Torch GPU)
runtimes_all_th2_gpu = zeros(length(sizes), length(rounding_modes), num_runs);
% Initialize average runtime storage (after discarding first run) for pychop
runtimes_avg_th2_gpu = zeros(length(sizes), length(rounding_modes));

% Set chop options for half precision
options.format = 'b'; % bhalf precision (bf16)
options.subnormal = 1; % Support subnormal numbers (optional, set to 0 to flush to zero)


% Loop over matrix sizes
for i = 1:length(sizes)
    n = sizes(i);
    fprintf('Testing matrix size: %d x %d\n', n, n);
    
    rng(i);
    % Generate random matrix (single precision input as required by chop)
    A = rand(n, n); % single(rand(n, n));
    % save(strcat(strcat("data/random/A", string(i)), ".mat"), "A");
    % save(strcat(strcat("data/random/A_b", string(i)), ".mat"), "A", "-v7.3");
    % Loop over rounding modes

    for j = 1:length(rounding_modes)
        options.round = rounding_modes(j);
        % ch = pc.Chop('h', rmode=j); % Same setting with chop
        
        fprintf('  Rounding mode: %s\n', mode_names{j});
        
        % Run 11 times
        for k = 1:num_runs
            A_np = np.array(A);

            pc.backend('numpy');
            ch = pc.LightChop(exp_bits=8, sig_bits=7, rmode=j);
            ch2 = pc.Chop('h', rmode=j);

            tic;
            A_chopped_th = ch(A_np); 
            runtimes_all_np(i, j, k) = toc;

            tic;
            A_chopped_th = ch2(A_np);
            runtimes_all_np2(i, j, k) = toc;

            pc.backend('torch');
            ch = pc.LightChop(exp_bits=5, sig_bits=10, rmode=j);
            ch2 = pc.Chop('h', rmode=j);

            A_np = th.from_numpy(A_np); % torch array
            
            tic;
            A_chopped_th = ch(A_np);
            runtimes_all_th(i, j, k) = toc;
            
            tic;
            A_chopped_th = ch2(A_np);
            runtimes_all_th2(i, j, k) = toc;
            
            A_np = A_np.to('cuda');

            tic;
            A_chopped_th = ch(A_np);
            runtimes_all_th_gpu(i, j, k) = toc;
            
            tic;
            A_chopped_th = ch2(A_np);
            runtimes_all_th2_gpu(i, j, k) = toc;

            tic;
            A_chopped = chop(A, options);
            runtimes_all(i, j, k) = toc;

        end

        % Compute average of runs 2 through 11 (discard first run)
        runtimes_avg(i, j) = mean(runtimes_all(i, j, 2:end));

        runtimes_avg_np(i, j) = mean(runtimes_all_np(i, j, 2:end));
        runtimes_avg_th(i, j) = mean(runtimes_all_th(i, j, 2:end));
        runtimes_avg_th_gpu(i, j) = mean(runtimes_all_th_gpu(i, j, 2:end));

        runtimes_avg_np2(i, j) = mean(runtimes_all_np2(i, j, 2:end));
        runtimes_avg_th2(i, j) = mean(runtimes_all_th2(i, j, 2:end));
        runtimes_avg_th2_gpu(i, j) = mean(runtimes_all_th2_gpu(i, j, 2:end));
    end
end

% Display results
disp('Average Runtimes (seconds, discarding first run):');
fprintf('Size\t%s\t%s\t%s\t%s\t%s\n', mode_names{:});
for i = 1:length(sizes)
    fprintf('%d\t', sizes(i));
    fprintf('%.6f\t', runtimes_avg(i, :));
    fprintf('\n');
end

%%% matlab data
save('results/chop_runtimes_avg_b.mat', 'sizes', 'rounding_modes', 'mode_names', 'runtimes_all', 'runtimes_avg');

% Create a table with sizes as the first column and runtimes for each mode as subsequent columns
csv_data = [sizes', runtimes_avg];
header = ['Size', mode_names]; % Cell array for column names

T = array2table(csv_data, 'VariableNames', header);
writetable(T, 'results/chop_runtimes_avg_b.csv');

disp('Results saved to chop_runtimes_avg_b.csv');

%%% numpy data
save('results/chop_runtimes_avg_np_b.mat', 'sizes', 'rounding_modes', 'mode_names', 'runtimes_all', 'runtimes_avg_np');

csv_data = [sizes', runtimes_avg_np];
header = ['Size', mode_names];

T = array2table(csv_data, 'VariableNames', header);
writetable(T, 'results/chop_runtimes_avg_np_b.csv');

disp('Results saved to chop_runtimes_avg_np_b.csv');

%%% torch data
save('results/chop_runtimes_avg_th_b.mat', 'sizes', 'rounding_modes', 'mode_names', 'runtimes_all', 'runtimes_avg_th');

csv_data = [sizes', runtimes_avg_th];
header = ['Size', mode_names]; 

T = array2table(csv_data, 'VariableNames', header);
writetable(T, 'results/chop_runtimes_avg_th_b.csv');

disp('Results saved to chop_runtimes_avg_th_b.csv');

%%% torch data gpu
save('results/chop_runtimes_avg_th_gpu_b.mat', 'sizes', 'rounding_modes', 'mode_names', 'runtimes_all', 'runtimes_avg_th');

csv_data = [sizes', runtimes_avg_th_gpu]; 
header = ['Size', mode_names]; 

T = array2table(csv_data, 'VariableNames', header);
writetable(T, 'results/chop_runtimes_avg_th_gpu_b.csv');

disp('Results saved to chop_runtimes_avg_th_gpu_b.csv');


%%% numpy data 2
save('results/chop_runtimes_avg_np2_b.mat', 'sizes', 'rounding_modes', 'mode_names', 'runtimes_all', 'runtimes_avg_np2');

csv_data = [sizes', runtimes_avg_np2];
header = ['Size', mode_names];

T = array2table(csv_data, 'VariableNames', header);
writetable(T, 'results/chop_runtimes_avg_np2_b.csv');

disp('Results saved to chop_runtimes_avg_np2_b.csv');

%%% torch data 2
save('results/chop_runtimes_avg_th2_b.mat', 'sizes', 'rounding_modes', 'mode_names', 'runtimes_all', 'runtimes_avg_th2');

csv_data = [sizes', runtimes_avg_th2];
header = ['Size', mode_names]; 

T = array2table(csv_data, 'VariableNames', header);
writetable(T, 'results/chop_runtimes_avg_th2_b.csv');

disp('Results saved to chop_runtimes_avg_th2_b.csv');

%%% torch data 2 gpu
save('results/chop_runtimes_avg_th2_gpu_b.mat', 'sizes', 'rounding_modes', 'mode_names', 'runtimes_all', 'runtimes_avg_th2');

csv_data = [sizes', runtimes_avg_th2_gpu];
header = ['Size', mode_names]; 

T = array2table(csv_data, 'VariableNames', header);
writetable(T, 'results/chop_runtimes_avg_th2_gpu_b.csv');

disp('Results saved to chop_runtimes_avg_th2_gpu_b.csv');
