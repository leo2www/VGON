clc;
clear;

% Set up environment
Path = pwd;
num_bins = 160;
data_path = 'results/DatainPaper_10000_1.mat';
load(data_path);  % Loads variable "gap"

% Prepare data
VGON = abs(gap);
minVGON = min(VGON);
maxVGON = max(VGON);
bin_edges = linspace(minVGON, maxVGON, num_bins + 1);
hist_counts = histcounts(VGON, bin_edges);

% Create plot
h = figure;
tiledlayout(1,1, 'TileSpacing', 'tight');
nexttile;

% Histogram-style scatter plot
bin_centers = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
sz = 10 * ones(1, num_bins);
sz(end-1:end) = [50, 60];  % Highlight last two

scatter(bin_centers, hist_counts, sz, 'filled', 'MarkerFaceColor', '#38E3FF');
hold on;

% Add threshold markers
add_threshold(VGON, 0.07, '#AB2A3C', 'r-.');
add_threshold(VGON, 0.08, '#324675', 'r:');

% Add max gap line
max_label = sprintf("Max gap = %.3f", maxVGON);
xline(maxVGON, 'r--', {max_label}, 'FontSize', 10, ...
    'Color', '#660874', 'LineWidth', 2, ...
    'LabelVerticalAlignment', 'bottom', ...
    'LabelHorizontalAlignment', 'left');

% Axes and labels
xlim([0.055, 0.085]);
ylim([0, 11000]);
xlabel('Gap');
ylabel('Number of States');
box on;
grid on;
ax = gca;
ax.TitleHorizontalAlignment = 'left';

% Export figure
output_name = fullfile(Path, 'results', 'gap_distribution');
print(h, output_name, '-dpng', '-r0');

%% Helper Function
function add_threshold(data, threshold, color, style)
    count = sum(data >= threshold);
    percent = count / length(data) * 100;
    label = {sprintf('Gap = %.3f', threshold), sprintf('%.2f%%', percent)};
    xline(threshold, style, label, ...
        'FontSize', 10, 'Color', color, 'LineWidth', 2, ...
        'LabelVerticalAlignment', 'middle', ...
        'LabelHorizontalAlignment', 'center');
end
