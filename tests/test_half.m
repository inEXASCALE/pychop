addpath("chop");
file = fopen("data1.txt");
values = fscanf(file, '%f');

%% half precision rounding to nearest ties to even and subnormal=0 
delete 'half/half_rmode_1_subnormal_0.txt';

options.format = 'h';
options.round = 1;
options.subnormal = 0;
chop([],options)

for i=1:10
    emu_val = chop(values(i))
    lines = string(emu_val);
    filename = "half/half_rmode_1_subnormal_0.txt";
    writelines(lines, filename, WriteMode="append")
end

%% half precision round up and subnormal=0 
delete 'half/half_rmode_2_subnormal_0.txt';

options.format = 'h';
options.round = 2;
options.subnormal = 0;
chop([],options)


for i=1:10
    emu_val = chop(values(i))
    lines = string(emu_val);
    filename = "half/half_rmode_2_subnormal_0.txt";
    writelines(lines, filename, WriteMode="append")
end



%% half precision rounding to nearest ties to even and subnormal=0 
delete 'half/half_rmode_3_subnormal_0.txt';

options.format = 'h';
options.round = 1;
options.subnormal = 0;
chop([],options)

for i=1:10
    emu_val = chop(values(i))
    lines = string(emu_val);
    filename = "half/half_rmode_3_subnormal_0.txt";
    writelines(lines, filename, WriteMode="append")
end

%% half precision round up and subnormal=0 
delete 'half/half_rmode_3_subnormal_0.txt';

options.format = 'h';
options.round = 3;
options.subnormal = 0;
chop([],options)

for i=1:10
    emu_val = chop(values(i))
    lines = string(emu_val);
    filename = "half/half_rmode_3_subnormal_0.txt";
    writelines(lines, filename, WriteMode="append")
end

%% half precision round up and subnormal=0 
delete 'half/half_rmode_4_subnormal_0.txt';

options.format = 'h';
options.round = 4;
options.subnormal = 0;
chop([],options)

for i=1:10
    emu_val = chop(values(i))
    lines = string(emu_val);
    filename = "half/half_rmode_4_subnormal_0.txt";
    writelines(lines, filename, WriteMode="append")
end


%% half precision rounding to nearest ties to even and subnormal=1 
delete 'half/half_rmode_1_subnormal_1.txt';

options.format = 'h';
options.round = 1;
options.subnormal = 0;
chop([],options)

for i=1:10
    emu_val = chop(values(i))
    lines = string(emu_val);
    filename = "half/half_rmode_1_subnormal_1.txt";
    writelines(lines, filename, WriteMode="append")
end

%% half precision round up and subnormal=1 
delete 'half/half_rmode_2_subnormal_1.txt';

options.format = 'h';
options.round = 2;
options.subnormal = 1;
chop([],options)


for i=1:10
    emu_val = chop(values(i))
    lines = string(emu_val);
    filename = "half/half_rmode_2_subnormal_1.txt";
    writelines(lines, filename, WriteMode="append")
end



%% half precision rounding to nearest ties to even and subnormal=1
delete 'half/half_rmode_3_subnormal_1.txt';

options.format = 'h';
options.round = 1;
options.subnormal = 1;
chop([],options)

for i=1:10
    emu_val = chop(values(i))
    lines = string(emu_val);
    filename = "half/half_rmode_3_subnormal_1.txt";
    writelines(lines, filename, WriteMode="append")
end

%% half precision round up and subnormal=0 
delete 'half/half_rmode_3_subnormal_1.txt';

options.format = 'h';
options.round = 3;
options.subnormal = 1;
chop([],options)

for i=1:10
    emu_val = chop(values(i))
    lines = string(emu_val);
    filename = "half/half_rmode_3_subnormal_1.txt";
    writelines(lines, filename, WriteMode="append")
end

%% half precision round up and subnormal=0 
delete 'half/half_rmode_4_subnormal_1.txt';

options.format = 'h';
options.round = 4;
options.subnormal = 1;
chop([],options)

for i=1:10
    emu_val = chop(values(i))
    lines = string(emu_val);
    filename = "half/half_rmode_4_subnormal_1.txt";
    writelines(lines, filename, WriteMode="append")
end

