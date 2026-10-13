% Generates a label image with bin/slicmex.mex.
% Usage:
%   octave --quiet --no-gui src/octave/script.m input output [n] [compactness]

args = argv();
if numel(args) < 2 || numel(args) > 4
  error('Usage: script.m input output [n] [compactness]');
endif

input_path = args{1};
output_path = args{2};
num_superpixels = 500;
compactness = 10;

if numel(args) >= 3
  num_superpixels = str2double(args{3});
endif
if numel(args) >= 4
  compactness = str2double(args{4});
endif

if !isfinite(num_superpixels) || num_superpixels < 1 || ...
   !isfinite(compactness) || compactness <= 0
  error('n must be positive and compactness must be greater than zero');
endif

script_dir = fileparts(mfilename('fullpath'));
repo_dir = fileparts(fileparts(script_dir));
addpath(fullfile(repo_dir, 'bin'));
pkg load image;

img = imread(input_path);
if !isa(img, 'uint8')
  error('slicmex.mex requires an 8-bit image');
endif

[labels, quantidade] = slicmex(img, round(num_superpixels), compactness);

% slicmex labels are zero-based. PNG stores labels + 1, preserving every ID.
imwrite(uint16(labels) + 1, output_path);
printf('Generated: %s\n', output_path);
printf('Final superpixels: %d\n', quantidade);
