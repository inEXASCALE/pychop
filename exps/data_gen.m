% Script to test chop with half precision, multiple runs, and averaging
clear all;
clc;
% torch.cuda.is_available()

addpath("../tests/chop")
pe = pyenv()
% Define matrix sizes
rng(0);

sizes = [2000, 4000, 6000, 8000, 10000]; % original: [128, 512, 2048, 8192, 32768] 

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
options.format = 'h'; % Half precision (fp16)
options.subnormal = 1; % Support subnormal numbers (optional, set to 0 to flush to zero)


% Loop over matrix sizes
for i = 1:length(sizes)
    n = sizes(i);
    fprintf('Testing matrix size: %d x %d\n', n, n);
    
    rng(i);
    % Generate random matrix (single precision input as required by chop)
    A = rand(n, n); % single(rand(n, n));
    % save(strcat(strcat("data/random/A", string(i)), ".mat"), "A");
    save(strcat(strcat("data/random/A", string(i)), ".mat"), "A", "-v7.3");
    % Loop over rounding modes

end
